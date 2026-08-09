#include "octree.h"

NAMESPACE_BEGIN(krr)

KRR_HOST void Octree::setMaximumMemory(uint bytes) {
	// sizeof(Node) = 9248, the same as mitusba version
	mConfiguration.mMaxNodeCount = bytes / sizeof(Node);
}

KRR_HOST uint Octree::getMaximumMemory() { return mConfiguration.mMaxNodeCount * sizeof(Node); }

KRR_HOST Octree::Octree(const uint bytes, const bool keepLastIterationStatistics,
						const bool cacheRadianceMeanAndCost) {
	if (cacheRadianceMeanAndCost) {
		mConfiguration.mSamplingNodeCacheCostAndRadianceMean = true;
	}

	auto stream = gpContext->cudaStream;
	cudaMallocAsync(&mAABB, sizeof(AABB), stream);
	cudaMallocAsync(&mNodesIndex, sizeof(uint), stream);
	setMaximumMemory(bytes);
	const long oneMoreNode = sizeof(Node); // 1 more for mNodeForInitialization
										   // the last one in mNodes

	const long realBytes = getMaximumMemory();
	// cudaMallocAsync(&mChildNodes, mConfiguration.mMaxNodeCount, stream);
	cudaMallocAsync(&mNodes, realBytes + oneMoreNode, stream);
	mNodeForInitialization = (Node *) (((char *) mNodes) + realBytes);

	mConfiguration.mLeafDecay = keepLastIterationStatistics ? 1.0f : 0.0f;

	Log(Info, "[Octree] Initializing Octree, Max Nodes = %d, Keep Last Iteration Statistics: %d",
		mConfiguration.mMaxNodeCount, keepLastIterationStatistics);
}

KRR_HOST void Octree::initialize(const AABB &aabb, const int depthInit) {
	auto stream = gpContext->cudaStream;

	GPUCall(
		[=] KRR_DEVICE() {
			// stack overflow if we just call: *mNodeForInitialization = Node();
			// manually initialize
			for (int i = 0; i < STRATUM_COUNT; ++i) {
				mNodeForInitialization->mChildren[i] = Octree::Node::Child();
				mNodeForInitialization->mFatherIndex = 0;
			}

			// initialize some GPU vals
			*mAABB		 = aabb;
			*mNodesIndex = 0; // initialization is sequential, so atomic is not needed here

			// initialize tree to some depth
			const float nodesWeightInit =
				mConfiguration.mMinimumLeafWeightForSampling * (1 << (3 * (depthInit - 1)));
			int newIdx = requestNode();
			Node *node = getNodeByIndex(newIdx);

			for (int stratum = 0; stratum < STRATUM_COUNT; ++stratum) {
				int newChildIndex = splitNodeIfNecessary(nodesWeightInit);
				if (newChildIndex != INVALID_NODE_INDEX) {
					node->mChildren[stratum].mIndex = newChildIndex;
					Node *child						= getNodeByIndex(newChildIndex);
					child->mFatherIndex				= newIdx;
				} else {
					break;
				}
			}
			// default: 73
			printf("[CUDA][Octree] nodes: %u/%u\n", *mNodesIndex, mConfiguration.mMaxNodeCount);
		},
		stream);
}

KRR_HOST Octree::~Octree() {
	auto stream = gpContext->cudaStream;
	cudaStreamSynchronize(stream);
	if (mNodes) {
		cudaFree(mNodes);
	}
	if (mAABB) {
		cudaFree(mAABB);
	}
	if (mNodesIndex) {
		cudaFree(mNodesIndex);
	}
}

KRR_HOST void Octree::renderUI() {
	ui::PushID("EARS::Octree");

	// if (ui::TreeNodeEx("Octree Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
	if (ui::TreeNodeEx("Octree Debug")) {

		auto stream = gpContext->cudaStream;

		if (ui::Button("Show Father<->Child Links")) {
			cudaStreamSynchronize(stream);
			GPUCall(
				[=] KRR_DEVICE() {
					for (int i = 0; i < *mNodesIndex; ++i) {
						Node *node = getNodeByIndex(i);
						for (int j = 0; j < STRATUM_COUNT; ++j) {
							Node::Child &child = node->mChildren[j];
							if (!child.isLeaf()) {
								printf("F->C->F = (%d, %d, %d)\n", i, child.mIndex,
									   getNodeByIndex(child.mIndex)->mFatherIndex);
							} else {
								printf("F->C = (%d, %d)\n", i, child.mIndex);
							}
						}
					}
				},
				stream);
			cudaStreamSynchronize(stream);
		}

		if (ui::Button("Clear Inner Nodes")) {
			cudaStreamSynchronize(stream);
			uint size;
			cudaMemcpy(&size, mNodesIndex, sizeof(uint), cudaMemcpyDeviceToHost); // sync
			clearInnerNodes(size);
		}

		if (ui::Button("Accumulate to Father")) {
			cudaStreamSynchronize(stream);
			uint size;
			cudaMemcpy(&size, mNodesIndex, sizeof(uint), cudaMemcpyDeviceToHost); // sync
			accumulateToFather(size);
		}

		if (ui::Button("Split Leaf Nodes")) {
			cudaStreamSynchronize(stream);
			uint size;
			cudaMemcpy(&size, mNodesIndex, sizeof(uint), cudaMemcpyDeviceToHost); // sync
			splitLeafNodes(size);
		}

		if (ui::Button("update Sampling Nodes")) {
			cudaStreamSynchronize(stream);
			uint size;
			cudaMemcpy(&size, mNodesIndex, sizeof(uint), cudaMemcpyDeviceToHost); // sync
			updateSamplingNodes(size);
		}

		ui::TreePop();
	}
	ui::PopID();
}

KRR_HOST void Octree::refine(bool needsSplitting) {
	auto stream = gpContext->cudaStream;
	cudaStreamSynchronize(stream);

	uint size;
	cudaMemcpy(&size, mNodesIndex, sizeof(uint), cudaMemcpyDeviceToHost); // sync

	if (OCTREE_DEBUG_ON) {
		printf("[Octree::refine] size = %u/%u\n", size, mConfiguration.mMaxNodeCount);
		printf("[Octree::refine] before accumulateToFather()\n");
		debug(nullptr);
	}

	// divide into 3 passes, so this can be parallelized(fit GPU)
	clearInnerNodes(size); // size matches
	accumulateToFather(size);

	if (OCTREE_DEBUG_ON) {
		printf("[Octree::refine] before splitLeafNodes()\n");
		debug(nullptr);
	}

	if (needsSplitting) {
		splitLeafNodes(size);
	}

	updateSamplingNodes(size);

	GPUCall(
		[=] KRR_DEVICE() {
			printf("[CUDA][Octree] nodes: %u/%u\n", *mNodesIndex, mConfiguration.mMaxNodeCount);
		},
		stream);
}

KRR_HOST void Octree::clearInnerNodes(uint size) {
	auto stream = gpContext->cudaStream;

	// the basic thread element is 1 child of the Node
	GPUParallelFor(
		size * STRATUM_COUNT,
		[=] KRR_DEVICE(const uint tid) mutable {
			const uint nodeIdx = tid / STRATUM_COUNT;
			const uint stratum = tid % STRATUM_COUNT;

			Node *node				= getNodeByIndex(nodeIdx);
			Node::Child &innerChild = node->mChildren[stratum];
			if (innerChild.isLeaf()) {
				// leave node
				return;
			}

			// clear
			for (int i = 0; i < BIN_COUNT; ++i) {
				innerChild.mTraining[i].decay(0.0f);
			}
		},
		stream);
}

KRR_HOST void Octree::accumulateToFather(uint size) {
	auto stream = gpContext->cudaStream;

	// the basic thread element is 1 child of the Node
	GPUParallelFor(
		size * STRATUM_COUNT,
		[=] KRR_DEVICE(const uint tid) mutable {
			const uint nodeIdx = tid / STRATUM_COUNT;
			const uint stratum = tid % STRATUM_COUNT;

			Node *node			   = getNodeByIndex(nodeIdx);
			Node::Child &leafChild = node->mChildren[stratum];
			if (!leafChild.isLeaf()) {
				// inner node
				return;
			}

			const TrainingNode *trainingNode = leafChild.mTraining;
			uint currentNodeIndex			 = nodeIdx;

			// accumulate
			// use do..while.. as
			// [1] leaf node can not be root node
			// [2] we should accumulate to the root
			do {
				Node *father		= getNodeByIndex(node->mFatherIndex);
				const uint childIdx = father->getChildStratum(currentNodeIndex);
				Node::Child &child	= father->mChildren[childIdx];

				for (int i = 0; i < BIN_COUNT; ++i) {
					atomicAdd(&child.mTraining[i].mLrWeight, trainingNode[i].mLrWeight);
					atomicAddColor(&child.mTraining[i].mLrFirstMoment,
								   trainingNode[i].mLrFirstMoment);

					atomicAddColor(&child.mTraining[i].mLrSecondMoment,
								   trainingNode[i].mLrSecondMoment);
					atomicAdd(&child.mTraining[i].mLrCost, trainingNode[i].mLrCost);
				}

				// up to the father
				currentNodeIndex = node->mFatherIndex;
				node			 = father;
			} while (node != mNodes); // root node
		},
		stream);
}

KRR_HOST void Octree::splitLeafNodes(uint size) {
	auto stream = gpContext->cudaStream;

	// the basic thread element is 1 child of the Node
	GPUParallelFor(
		size * STRATUM_COUNT,
		[=] KRR_DEVICE(const uint tid) mutable {
			const uint nodeIdx = tid / STRATUM_COUNT;
			const uint stratum = tid % STRATUM_COUNT;

			Node *node			   = getNodeByIndex(nodeIdx);
			Node::Child &leafChild = node->mChildren[stratum];
			if (!leafChild.isLeaf()) {
				// inner node
				return;
			}

			// split
			uint newChildIndex = splitNodeIfNecessary((float) leafChild.maxTrainingWeight());
			if (newChildIndex != INVALID_NODE_INDEX) {
				node->mChildren[stratum].mIndex = newChildIndex;
				Node *child						= getNodeByIndex(newChildIndex);
				child->mFatherIndex				= nodeIdx;
			}
		},
		stream);
}

KRR_HOST void Octree::updateSamplingNodes(uint size) {
	auto stream = gpContext->cudaStream;

	GPUParallelFor(
		size * STRATUM_COUNT,
		[=] KRR_DEVICE(const uint tid) mutable {
			const uint nodeIdx = tid / STRATUM_COUNT;
			const uint stratum = tid % STRATUM_COUNT;

			Node *node		   = getNodeByIndex(nodeIdx);
			Node::Child &child = node->mChildren[stratum];

			for (int bin = 0; bin < BIN_COUNT; ++bin) {
				child.mSampling[bin].learnFrom(child.mTraining[bin], mConfiguration);
				child.mTraining[bin].decay(mConfiguration.mLeafDecay);
			}
		},
		stream);
}

KRR_HOST void Octree::debug(void *everything) {
	auto stream = gpContext->cudaStream;
	cudaStreamSynchronize(stream);

	GPUCall(
		[=] KRR_DEVICE() {
			printf("[CUDA][Octree] nodes: %u/%u\n", *mNodesIndex, mConfiguration.mMaxNodeCount);

			// check all nodes, print statistics
			TrainingNode t1, t2;
			t1.decay(0);
			t2.decay(0);

			for (uint i = 0; i < *mNodesIndex; ++i) {
				Node *node = getNodeByIndex(i);
				for (int j = 0; j < STRATUM_COUNT; ++j) {
					Node::Child &child = node->mChildren[j];
					for (int k = 0; k < BIN_COUNT; ++k) {
						TrainingNode *trainingNode = child.mTraining + k;
						if (trainingNode->mLrWeight > t1.mLrWeight) {
							t1 = *trainingNode;
						}
						if (trainingNode->getLrEstimate().mean() > t2.getLrEstimate().mean()) {
							t2 = *trainingNode;
						}
					}
				}
			}

			printf("[Octree] Max Weight\n");
			t1.printInfos();

			printf("[Octree] Max LrEstimate\n");
			t2.printInfos();
		},
		stream);

	cudaStreamSynchronize(stream);
}

NAMESPACE_END(krr)

/*  This code should be run on host side.
	All the data manipulated by this code should reside on host memory. */
#include <algorithm>
#include <execution>

#include "tree.h"

/*	Max d-tree depth: 20; max d-tree nodes: 2e16.
 *	Max s-tree nodes: 2e32; max s-tree depth is unlimited.
 */

#include "treenodetypes.h"

NAMESPACE_BEGIN(krr)

template <typename T> KRR_HOST void QuadTreeNode<T>::initialize() {
	for (size_t i = 0; i < 4 /*m_sum.size()*/; ++i) {
		mChildren[i] = 0;
		mSum[i].store(0);
	}
}

template <typename T> KRR_HOST void QuadTreeNode<T>::copyFrom(const QuadTreeNode<T> &arg) {
	for (int i = 0; i < 4; ++i) {
		setSum(i, arg.sumAll(i));
		mChildren[i] = arg.mChildren[i];
	}
}

template <typename T> KRR_HOST QuadTreeNode<T>::QuadTreeNode(const QuadTreeNode<T> &arg) {
	copyFrom(arg);
}

template <typename T>
KRR_HOST QuadTreeNode<T> &QuadTreeNode<T>::operator=(const QuadTreeNode<T> &arg) {
	copyFrom(arg);
	return *this;
}

/*	Ensure that each quadtree node's sum of irradiance estimates equals that of all its children.
	This function do not change the topology of the D-Tree.
	@modified VAPG  */
template <typename T>
KRR_HOST void QuadTreeNode<T>::build(std::vector<QuadTreeNode<T>> &nodes,
									 EDistribution distribution,
									 float parentSize) { // [called by host]

	float childSize = parentSize / 4.f;
	for (int i = 0; i < 4; ++i) {
		// During sampling, all irradiance estimates are accumulated in
		// the leaves, so the leaves are built by definition.
		if (isLeaf(i)) {
			if (distribution == EDistribution::ESimple || distribution == EDistribution::EFull) {
				// @addition VAPG
				// pointwise square-root operation since guiding distribution based on second moment
				setSum(i, (sumAll(i) * childSize).sqrtf());
			}
			continue;
		}

		QuadTreeNode &c = nodes[child(i)];

		// Recursively build each child such that their sum becomes valid...
		c.build(nodes, distribution, childSize);

		// ...then sum up the children's sums.
		T sum = 0;
		for (int j = 0; j < 4; ++j) {
			sum += c.sumAll(j);
		}
		setSum(i, sum);
	}
}

template <typename T> KRR_HOST void DTree<T>::initialize() {
	mSum.store(0);
	mStatisticalWeight.store(0);
	mMaxDepth = 0;
	std::vector<QuadTreeNode<T>> nodes(1);
	nodes.front().initialize();
	mNodes.alloc_and_copy_from_host(nodes);
	CUDA_SYNC_CHECK();
}

template <typename T> KRR_HOST void DTree<T>::clear() {
	CUDA_SYNC(mNodes.clear());
	mSum.store(0);
	mStatisticalWeight.store(0);
	mMaxDepth = 0;
}

template <typename T> KRR_HOST DTree<T> &DTree<T>::operator=(const DTree<T> &other) {
	mNodes	  = other.mNodes; /* Deep copies the D-Tree nodes. */
	mMaxDepth = other.mMaxDepth;
	mSum.store(other.mSum);
	mStatisticalWeight.store(other.mStatisticalWeight);
	return *this;
}

template <typename T> KRR_HOST DTree<T>::DTree(const DTree<T> &other) { *this = other; }

/* This function adaptively subdivides / prunes the D-Tree, recursively in a sequential manner. */
template <typename T>
KRR_HOST void DTree<T>::reset(const DTree<T> &previousDTree, int newMaxDepth,
							  float subdivisionThreshold) {
	struct StackNode {
		size_t nodeIndex;
		size_t otherNodeIndex;
		const std::vector<QuadTreeNode<T>> *nodes;
		int depth;
	};

	/* Do the adaptive subdivision on host-side. */
	std::vector<QuadTreeNode<T>> this_nodes(1);
	std::vector<QuadTreeNode<T>> other_nodes(previousDTree.mNodes.size());
	mSum.store(0);
	mStatisticalWeight.store(0);
	mMaxDepth = 0;
	this_nodes.back().initialize();
	previousDTree.mNodes.copy_to_host(other_nodes.data(), previousDTree.mNodes.size());

	std::stack<StackNode> nodeIndices;
	nodeIndices.push({0, 0, &other_nodes, 1});

	const float total = previousDTree.mSum.load();

	// Create the topology of the new DTree to be the refined version
	// of the previous DTree. Subdivision is recursive if enough energy is there.
	while (!nodeIndices.empty()) {
		StackNode sNode = nodeIndices.top();
		nodeIndices.pop();

		mMaxDepth = max(mMaxDepth, sNode.depth);

		for (int i = 0; i < 4; ++i) {
			const QuadTreeNode<T> &otherNode = (*sNode.nodes)[sNode.otherNodeIndex];
			/* This makes each d-tree have 85+ nodes (4 layers) even without any radiance records.
			 */
			const float fraction = total > 0 ? (otherNode.sum(i) / total) : pow(0.25f, sNode.depth);
			CHECK_LE(fraction, 1.f + M_EPSILON);

			if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
				if (!otherNode.isLeaf(i)) {
					CHECK_EQ(sNode.nodes, &other_nodes);
					nodeIndices.push(
						{this_nodes.size(), otherNode.child(i), &other_nodes, sNode.depth + 1});
				} else {
					nodeIndices.push(
						{this_nodes.size(), this_nodes.size(), &this_nodes, sNode.depth + 1});
				}

				this_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(this_nodes.size()));
				this_nodes.emplace_back();
				this_nodes.back().initialize();
				this_nodes.back().setSum(otherNode.sumAll(i) / 4);

				if (this_nodes.size() > std::numeric_limits<uint16_t>::max()) {
					logWarning("[ResetDTree] DTreeWrapper hit maximum children count (65536).");
					nodeIndices = std::stack<StackNode>();
					break;
				}
			}
		}
	}

	for (auto &node : this_nodes) { /* zeros all the radiance of nodes, rebuild them later. */
		node.resetZero();
	}

	/* Copy the processed host nodes to device side. */
	mNodes.alloc_and_copy_from_host(this_nodes);
	CUDA_SYNC_CHECK();
}

/* Make sure *this is on host memory now. This function would not change the topology of the D-Tree.
   @modified VAPG */
template <typename T> KRR_HOST void DTree<T>::build(EDistribution distribution) {
	size_t n_nodes = mNodes.size();
	std::vector<QuadTreeNode<T>> nodes(n_nodes);
	mNodes.copy_to_host(nodes.data(), n_nodes);

	QuadTreeNode<T> &root = nodes[0];
	// Build the quadtree recursively, starting from its root.
	root.build(nodes, distribution, 1.f);

	// Ensure that the overall sum of irradiance estimates equals
	// the sum of irradiance estimates found in the quadtree.
	T sum = 0;
	for (int i = 0; i < 4; ++i) {
		sum += root.sumAll(i);
	}
	mSum.store(sum);
	mNodes.copy_from_host(nodes.data(), n_nodes);
}

template <typename T> KRR_HOST void DTreeWrapper<T>::initialize() {
	CUDA_SYNC(mSampling.initialize());
	CUDA_SYNC(mBuilding.initialize());
}

template <typename T> KRR_HOST void DTreeWrapper<T>::clear() {
	CUDA_SYNC(mSampling.clear());
	CUDA_SYNC(mBuilding.clear());
}

template <typename T>
KRR_HOST DTreeWrapper<T> &DTreeWrapper<T>::operator=(const DTreeWrapper<T> &other) {
	mBuilding = other.mBuilding;
	mSampling = other.mSampling;
	return *this;
}

template <typename T> KRR_HOST void DTreeWrapper<T>::build(EDistribution distribution) {
	mBuilding.build(distribution);
	mSampling.clear();
	mSampling = mBuilding;
}

template <typename T>
KRR_HOST void DTreeWrapper<T>::reset(int maxDepth, float subdivisionThreshold) {
	mBuilding.reset(mSampling, maxDepth, subdivisionThreshold);
}

template <typename T> KRR_HOST void STreeNode<T>::initialize() {
	mIsLeaf = true;
	mDTree.initialize();
	CUDA_SYNC_CHECK();
}

template <typename T>
KRR_HOST void STreeNode<T>::forEachLeaf(
	std::function<void(const DTreeWrapper<T> *, const Vector3f &, const Vector3f &)> func,
	Vector3f p, Vector3f size, const TypedBuffer<STreeNode<T>> &nodes) const {

	if (mIsLeaf) {
		func(&mDTree, p, size);
	} else {
		size[mAxis] /= 2;
		for (int i = 0; i < 2; ++i) {
			Vector3f childP = p;
			if (i == 1) {
				childP[mAxis] += size[mAxis];
			}

			nodes[mChildren[i]].forEachLeaf(func, childP, size, nodes);
		}
	}
}

template <typename T> KRR_HOST STree<T>::STree(const AABB &aabb, Allocator alloc) {
	clear();
	mAabb = aabb;
	// Enlarge AABB to turn it into a cube. This has the effect of nicer hierarchical subdivisions.
	Vector3f size = mAabb.diagonal();
	mAabb.extend(mAabb.min() + Vector3f(size.maxCoeff()));
}

template <typename T> KRR_HOST void STree<T>::clear() {
	forEachDTreeWrapper([](DTreeWrapper<T> *dtree) {
		dtree->clear(); /* free the memories of the quadtrees */
	});
	std::vector<STreeNode<T>> nodes(1);
	nodes.front().initialize();
	mNodes.alloc_and_copy_from_host(nodes); // initialize the super root tree node
}

template <typename T> KRR_HOST void STree<T>::subdivideAll() {
	size_t n_nodes = mNodes.size();
	std::vector<STreeNode<T>> nodes(n_nodes);
	mNodes.copy_to_host(nodes.data(), n_nodes);
	for (int i = 0; i < n_nodes; ++i) {
		if (mNodes[i].mIsLeaf) {
			subdivide(i, nodes);
		}
	}
	mNodes.alloc_and_copy_from_host(nodes);
}

/* This is the actual function that directly changes the topology of the S-Tree. */
template <typename T>
KRR_HOST void STree<T>::subdivide(int nodeIdx, std::vector<STreeNode<T>> &nodes) {
	// Add 2 child nodes
	nodes.resize(nodes.size() + 2);

	if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
		logWarning("[SubdivideSTree] DTreeWrapper hit maximum children count.");
		return;
	}

	STreeNode<T> &cur = nodes[nodeIdx];
	for (int i = 0; i < 2; ++i) {
		uint32_t idx		= (uint32_t) nodes.size() - 2 + i;
		STreeNode<T> &child = nodes[idx];
		cur.mChildren[i]	= idx;
		child.initialize();
		child.mAxis	 = (cur.mAxis + 1) % 3;
		child.mDTree = cur.mDTree;
		child.mDTree.setStatisticalWeightBuilding(child.mDTree.statisticalWeightBuilding() / 2);
	}
	cur.mIsLeaf = false;
	cur.mDTree.clear(); // Reset to an empty dtree to save memory.
	CUDA_SYNC_CHECK();
}

template <typename T>
KRR_HOST void STree<T>::forEachDTreeWrapper(std::function<void(DTreeWrapper<T> *)> func) {
	int n_nodes = static_cast<int>(mNodes.size());
	Log(Info, "[ForEachDTreeWrapper] There are %d S-Tree nodes to process...", n_nodes);
	std::vector<STreeNode<T>> nodes(n_nodes);
	mNodes.copy_to_host(nodes.data(), n_nodes);

	std::for_each(std::execution::par, nodes.begin(), nodes.end(), [&func](STreeNode<T> &node) {
		if (node.mIsLeaf) {
			func(&node.mDTree);
		}
	});
	mNodes.copy_from_host(nodes.data(), nodes.size());
}

template <typename T>
KRR_HOST bool STree<T>::shallSplit(const STreeNode<T> &node, size_t samplesRequired) {
	// Log(Info, "The node at depth %d has a statistical weight of %f, with the specified required
	// sample is %zd", 	depth, node.dTree.statisticalWeightBuilding(), samplesRequired);
	return node.mDTree.statisticalWeightBuilding() > samplesRequired;
}

/*	Only this function would change the topology of the S-Tree, sequentially executed.
	To adaptively subdivision, should be run on hostcode? */
template <typename T> KRR_HOST void STree<T>::refine(size_t sTreeThreshold, int maxMB) {
	// These work are done on CPU!
	size_t n_nodes = mNodes.size();
	Log(Info, "[REFINE] There are %zd S-Tree nodes to traverse...", n_nodes);
	std::vector<STreeNode<T>> nodes(n_nodes);
	mNodes.copy_to_host(nodes.data(), n_nodes);

	if (maxMB >= 0) {
		size_t approxMemoryFootprint = 0;
		for (const auto &node : nodes) {
			approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
		}

		if (approxMemoryFootprint / 1000000 >= (size_t) maxMB) {
			return;
		}
	}

	struct StackNode {
		size_t index;
		int depth;
	};

	std::stack<StackNode> nodeIndices;
	nodeIndices.push({0, 1});
	while (!nodeIndices.empty()) {
		StackNode sNode = nodeIndices.top();
		nodeIndices.pop();
		// Log(Info, "Traversing the %zd-th S-Tree node at depth %d", sNode.index, sNode.depth);

		// Subdivide if needed and leaf
		if (nodes[sNode.index].mIsLeaf) {
			if (nodes.size() < std::numeric_limits<uint32_t>::max() &&
				shallSplit(nodes[sNode.index], sTreeThreshold)) {
				subdivide((int) sNode.index, nodes);
			}
		}

		// Add children to stack if we're not
		if (!nodes[sNode.index].mIsLeaf) {
			const STreeNode<T> &node = nodes[sNode.index];
			for (int i = 0; i < 2; ++i) {
				nodeIndices.push({node.mChildren[i], sNode.depth + 1});
			}
		}
	}

	/* Copy the host-side new S-Tree to device side */
	mNodes.alloc_and_copy_from_host(nodes);
	CUDA_SYNC_CHECK();
}

/* Collect the statistics for the sampling tree (not the building tree that get reset). */
template <typename T> KRR_HOST void STree<T>::gatherStatistics() const {
	cudaDeviceSynchronize();
	size_t n_nodes = mNodes.size();
	std::vector<STreeNode<T>> nodes(n_nodes);
	mNodes.copy_to_host(nodes.data(), n_nodes);

	int maxDepth			   = 0;
	int minDepth			   = std::numeric_limits<int>::max();
	float avgDepth			   = 0;
	float maxAvgRadiance	   = 0;
	float minAvgRadiance	   = std::numeric_limits<float>::max();
	float avgAvgRadiance	   = 0;
	size_t maxNodes			   = 0;
	size_t minNodes			   = std::numeric_limits<size_t>::max();
	float avgNodes			   = 0;
	float maxStatisticalWeight = 0;
	float minStatisticalWeight = std::numeric_limits<float>::max();
	float avgStatisticalWeight = 0;

	int nPoints		 = 0;
	int nPointsNodes = 0;

	for (const auto &node : nodes) {
		if (node.mIsLeaf) {
			/* These statistics are all gathered from the sampling tree. */
			const DTreeWrapper<T> *dTree = node.dTreeWrapper();
			const int depth				 = dTree->depth();
			maxDepth					 = max(maxDepth, depth);
			minDepth					 = min(minDepth, depth);
			avgDepth += depth;

			const float avgRadiance = dTree->meanRadiance();
			maxAvgRadiance			= max(maxAvgRadiance, avgRadiance);
			minAvgRadiance			= min(minAvgRadiance, avgRadiance);
			avgAvgRadiance += avgRadiance;

			if (dTree->numNodes() > 1) {
				const size_t cur_nodes = dTree->numNodes();
				maxNodes			   = max(maxNodes, cur_nodes);
				minNodes			   = min(minNodes, cur_nodes);
				avgNodes += cur_nodes;
				++nPointsNodes;
			}

			const float statisticalWeight = dTree->statisticalWeight();
			maxStatisticalWeight		  = max(maxStatisticalWeight, statisticalWeight);
			minStatisticalWeight		  = min(minStatisticalWeight, statisticalWeight);
			avgStatisticalWeight += statisticalWeight;
			++nPoints;
		}
	}

	if (nPoints > 0) {
		avgDepth /= nPoints;
		avgAvgRadiance /= nPoints;
		if (nPointsNodes > 0) {
			avgNodes /= nPointsNodes;
		}
		avgStatisticalWeight /= nPoints;
	}

	printf("Distribution statistics:\n"
		   "  Depth of D-Trees         = [%d, %f, %d]\n"
		   "  Mean radiance            = [%f, %f, %f]\n"
		   "  Node count of D-Trees    = [%d, %f, %d]\n"
		   "  Stat. weight	of D-Trees  = [%f, %f, %f]\n",
		   minDepth, avgDepth, maxDepth, minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
		   static_cast<int>(minNodes), avgNodes, static_cast<int>(maxNodes), minStatisticalWeight,
		   avgStatisticalWeight, maxStatisticalWeight);
}

// template declarations
// [1] PPG
template class QuadTreeNode<SimpleTreeNode>;
template class STreeNode<SimpleTreeNode>;
template class DTree<SimpleTreeNode>;
template class STree<SimpleTreeNode>;
template class DTreeWrapper<SimpleTreeNode>;

// [2] ADRRS
template class QuadTreeNode<RadianceTreeNode>;
template class STreeNode<RadianceTreeNode>;
template class DTree<RadianceTreeNode>;
template class STree<RadianceTreeNode>;
template class DTreeWrapper<RadianceTreeNode>;

NAMESPACE_END(krr)
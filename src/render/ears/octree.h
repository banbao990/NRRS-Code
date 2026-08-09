#pragma once
#include <atomic>
#include <array>
#include <stack>

#include "common.h"
#include "sampler.h"
#include "logger.h"
#include "window.h"
#include "device/gpustd.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/atomic.h"

#include "util/check.h"
NAMESPACE_BEGIN(krr)

// [ATTENTION] Octree should be allocated by cudaManagedMalloc()
class Octree {
public:
	static constexpr int HISTOGRAM_RESOLUTION = 4;
	static constexpr int BIN_COUNT			  = HISTOGRAM_RESOLUTION * HISTOGRAM_RESOLUTION;
	static constexpr int STRATUM_COUNT		  = 8;
	static constexpr int INVALID_NODE_INDEX	  = -1;
	static constexpr bool OCTREE_DEBUG_ON	  = true;

	// debug
	static constexpr int DEBUG_TID = 1315;
	template <typename... Args> KRR_DEVICE void debugPrint(const char *fmt, Args &&...args) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid == DEBUG_TID) {
			printf(fmt, std::forward<Args>(args)...);
		}
	}

	KRR_DEVICE void atomicAddColor(const RGB *addr, const RGB &val) {
		float *addrFloat = (float *) addr;
		atomicAdd(addrFloat, val[0]);
		atomicAdd(addrFloat + 1, val[1]);
		atomicAdd(addrFloat + 2, val[2]);
	}
	struct Node;

	struct Configuration {
		float mMinimumLeafWeightForSampling = 40000;
		float mMinimumLeafWeightForTraining = 20000;
		float mLeafDecay = 0; /// set to 0 for hard reset after an iteration, 1 for no reset at all
		uint mMaxNodeCount = 0;

		// true: LrEstimate: RGB: Raidance
		// false: LrEstimate: R: adianceMean; G: Cost; B: No use
		bool mSamplingNodeCacheCostAndRadianceMean{false};
	};

	struct TrainingNode {
		KRR_DEVICE void decay(float decayFactor) {
			mLrWeight *= (uint) decayFactor;
			mLrFirstMoment *= decayFactor;
			mLrSecondMoment *= decayFactor;
			mLrCost *= decayFactor;
		}

		KRR_DEVICE TrainingNode &operator+=(const TrainingNode &other) {
			mLrWeight += other.mLrWeight;
			mLrFirstMoment += other.mLrFirstMoment;
			mLrSecondMoment += other.mLrSecondMoment;
			mLrCost += other.mLrCost;
			return *this;
		}

		KRR_DEVICE uint getWeight() const { return mLrWeight; }

		KRR_DEVICE RGB getLrEstimate() const {
			return mLrWeight > 0 ? mLrFirstMoment / mLrWeight : RGB(0.0f);
		}

		KRR_DEVICE RGB getLrSecondMoment() const {
			return mLrWeight > 0 ? mLrSecondMoment / mLrWeight : RGB(0.0f);
		}

		KRR_DEVICE RGB getLrVariance() const {
			if (mLrWeight == 0) {
				return RGB(0.0f);
			}

			RGB result;
			for (int i = 0; i < 3; ++i) {
				result[i] =
					max(mLrSecondMoment[i] / mLrWeight - float(pow2(mLrFirstMoment[i] / mLrWeight)),
						float(0));
			}
			return result;
		}

		KRR_DEVICE float getLrCost() const {
			return mLrWeight > 0 ? ((float) mLrCost) / mLrWeight : 0;
		}

		KRR_DEVICE void atomicAddColor(const RGB *addr, const RGB &val) {
			float *addrFloat = (float *) addr;
			atomicAdd(addrFloat, val[0]);
			atomicAdd(addrFloat + 1, val[1]);
			atomicAdd(addrFloat + 2, val[2]);
		}

		KRR_DEVICE void splatLrEstimate(const RGB &sum, uint cost, uint samples,
										Octree::Node *trainingPosNode) {

			auto sum2 = sum * sum;
			atomicAddColor(&mLrFirstMoment, sum);
			atomicAddColor(&mLrSecondMoment, sum2);
			atomicAdd(&mLrCost, cost);
			atomicAdd(&mLrWeight, samples);
		}

		KRR_DEVICE void printInfos() const {
			const auto avgLrEstimate	 = getLrEstimate();
			const auto avgLrSecondMoment = getLrSecondMoment();
			const auto avgLrCost		 = getLrCost();
			printf("[Octree::TrainingNode]\n"
				   "\tLrCost: %u\n\tLrFirstMoment: (%f, %f, %f)\n"
				   "\tLrSecondMoment: (%f, %f, %f)\n\tLrWeight: %u\n"
				   "\tAverage LrCost: %f\n\tAverage LrFirstMoment: (%f, %f, %f)\n"
				   "\tAverage LrSecondMoment: (%f, %f, %f)\n",
				   mLrCost, mLrFirstMoment.x(), mLrFirstMoment.y(), mLrFirstMoment.z(),
				   mLrSecondMoment.x(), mLrSecondMoment.y(), mLrSecondMoment.z(), mLrWeight,
				   avgLrCost, avgLrEstimate.x(), avgLrEstimate.y(), avgLrEstimate.z(),
				   avgLrSecondMoment.x(), avgLrSecondMoment.y(), avgLrSecondMoment.z());
		}

		// TODO: make it private
		// private:
		uint mLrWeight{0u};
		RGB mLrFirstMoment{0.0f};
		RGB mLrSecondMoment{0.0f};
		uint mLrCost{0u};
	};

	struct SamplingNode {
		KRR_DEVICE bool isValid() const { return mIsValid; }
		KRR_DEVICE void learnFrom(const TrainingNode &trainingNode, const Configuration &config) {
			mIsValid = trainingNode.getWeight() >= config.mMinimumLeafWeightForSampling;

			if (trainingNode.getWeight() > 0) {
				mLrEstimate = trainingNode.getLrEstimate();
				if (config.mSamplingNodeCacheCostAndRadianceMean) {
					mLrEstimate[0] = mLrEstimate.mean();
					mLrEstimate[1] = trainingNode.getLrCost(); // only for 'earst'
				}

				if (trainingNode.getLrCost() > 0) {
					mEarsFactorR = trainingNode.getLrSecondMoment() / trainingNode.getLrCost();
					mEarsFactorS = trainingNode.getLrVariance() / trainingNode.getLrCost();
				} else {
					/// there can be caches where no work is done
					/// (e.g., failed strict normals checks meaning no NEE samples or BSDF samples
					/// are ever taken)
					mEarsFactorR = RGB(0.0f);
					mEarsFactorS = RGB(0.0f);
				}
			}
		}

		RGB mLrEstimate;
		RGB mEarsFactorR; // 2nd-moment / cost
		RGB mEarsFactorS; // variance / cost

	private:
		bool mIsValid;
	};

	struct Node {
		struct Child {
			int mIndex{INVALID_NODE_INDEX}; // index in Octree::mNodes
			TrainingNode mTraining[BIN_COUNT];
			SamplingNode mSampling[BIN_COUNT];

			KRR_DEVICE bool isLeaf() const { return mIndex == INVALID_NODE_INDEX; }
			KRR_DEVICE uint maxTrainingWeight() const {
				uint weight = 0u;
				for (const auto &t : mTraining) {
					weight = max(weight, t.getWeight());
				}
				return weight;
			}
		};

		KRR_DEVICE int getChildStratum(int childNodeIndex) const {
			for (int stratum = 0; stratum < STRATUM_COUNT; ++stratum) {
				if (mChildren[stratum].mIndex == childNodeIndex) {
					return stratum;
				}
			}
			printf("\033[33m[Warning] getChildStratum() Error: child node not found in father "
				   "node\033[0m\n");
			return -1;
		}

		// must manually initialize
		int mFatherIndex{0};
		Child mChildren[Octree::STRATUM_COUNT];
	};

private:
	Configuration mConfiguration;

	// GPU Data
	Node *mNodes{nullptr};
	uint *mNodesIndex{nullptr};
	AABB *mAABB{nullptr};
	// uint *mChildNodes{ nullptr };

	// temp data
	Node *mNodeForInitialization;

private:
	// [x, y, z]
	//  x -->
	//  y |
	//    V
	//   z = 0      z = 1
	// +---+---+  +---+---+
	// | 0 | 1 |  | 4 | 5 |
	// +---+---+  +---+---+
	// | 2 | 3 |  | 6 | 7 |
	// +---+---+  +---+---+
	KRR_DEVICE int stratumIndex(Vector3f &pos) {
		int index = 0;
		for (int dim = 0; dim < 3; ++dim) {
			int bit = pos[dim] >= 0.5f;
			index |= bit << dim;
			pos[dim] = pos[dim] * 2 - bit;
		}
		return index;
	}

	// no splitting: return INVALID_INDEX
	// you should link the father<->child relationship after call this function
	KRR_DEVICE uint splitNodeIfNecessary(float weight) {
		// TODO: needs optimization, recursive => non-recursive

		if (weight < mConfiguration.mMinimumLeafWeightForTraining) {
			/// splitting not necessary
			return INVALID_NODE_INDEX;
		}

		debugPrint("[Octree::splitLeafNodes]: weight = %f\n", weight);

		int newNodeIdx = requestNode();
		if (newNodeIdx == INVALID_NODE_INDEX) {
			/// we have already reached the maximum node number
			return INVALID_NODE_INDEX;
		}

		Node *newNode = getNodeByIndex(newNodeIdx);

		for (int stratum = 0; stratum < STRATUM_COUNT; ++stratum) {
			/// split recursively if needed
			uint newChildIdx = splitNodeIfNecessary(weight / STRATUM_COUNT);
			if (newChildIdx != INVALID_NODE_INDEX) {
				newNode->mChildren[stratum].mIndex = newChildIdx;
				Node *child						   = getNodeByIndex(newChildIdx);
				child->mFatherIndex				   = newNodeIdx;
			} else {
				break;
			}
		}

		return newNodeIdx;
	}

	KRR_DEVICE void build(uint index, bool needsSplitting, TrainingNode *sum) {
		for (int stratum = 0; stratum < STRATUM_COUNT; ++stratum) {
			// as we do not change mNodes's size, we can just use this &
			// cpp version cannot use this &, because the reference may changed when vector resize
			Node::Child &child = mNodes[index].mChildren[stratum];
			if (child.isLeaf()) {
				if (needsSplitting) {
					uint newChildIndex = splitNodeIfNecessary(child.maxTrainingWeight());
					child.mIndex	   = newChildIndex;
				}
			} else {
				// build recursively
				TrainingNode buildResult[BIN_COUNT];
				build(child.mIndex, needsSplitting, buildResult);
				for (int i = 0; i < BIN_COUNT; ++i) {
					child.mTraining[i] = buildResult[i];
				}
			}

			for (int bin = 0; bin < BIN_COUNT; ++bin) {
				sum[bin] += child.mTraining[bin];
				child.mSampling[bin].learnFrom(child.mTraining[bin], mConfiguration);
				child.mTraining[bin].decay(mConfiguration.mLeafDecay);
			}
		}
	}

	KRR_HOST void setMaximumMemory(uint bytes);
	KRR_HOST uint getMaximumMemory();

public:
	KRR_HOST Octree(const uint bytes, const bool keepLastIterationStatistics,
					const bool cacheRadianceMeanAndCost);
	KRR_HOST void initialize(const AABB &aabb, const int depthInit = 2);
	KRR_HOST ~Octree();

	KRR_HOST void renderUI();

	KRR_HOST void debug(void *everything);

	/**
	 * Accumulates all the data from training into the sampling nodes, refines the tree and resets
	 * the training nodes.
	 */
	KRR_HOST void refine(bool needsSplitting);
	KRR_HOST void clearInnerNodes(uint size);
	KRR_HOST void accumulateToFather(uint size);
	KRR_HOST void splitLeafNodes(uint size);
	KRR_HOST void updateSamplingNodes(uint size);

	// map to unit cube [0, 1]^3
	KRR_DEVICE Vector3f mapPointToUnitCube(const Vector3f &p) const {
		Vector3f size	= mAABB->diagonal();
		Vector3f result = (p - mAABB->min()).cwiseQuotient(size);
		return result;
	}

	// input should be normalized
	KRR_DEVICE Vector2f dirToCanonical(const Vector3f &d) const {
		if (d.hasInf()) {
			return Vector2f{0, 0};
		}
		const float cosTheta = min(max(d.z(), -1.0f), 1.0f);
		float phi			 = atan2(d.y(), d.x());
		while (phi < 0) {
			phi += 2.0 * M_PI;
		}

		return {(cosTheta + 1) / 2, phi / (2 * M_PI)};
	}

	KRR_DEVICE int mapOutgoingDirectionToHistogramBin(const Vector3f &wo) const {
		const Vector2f p = dirToCanonical(wo);
		const int res	 = Octree::HISTOGRAM_RESOLUTION;
		const int result = min(int(p.x() * res), res - 1) + min(int(p.y() * res), res - 1) * res;
		return result;
	}

	KRR_DEVICE int Octree::requestNode() {
		if (*mNodesIndex > mConfiguration.mMaxNodeCount) {
			return INVALID_NODE_INDEX;
		}

		uint oldVal = atomicAdd(mNodesIndex, 1u);
		if (oldVal >= mConfiguration.mMaxNodeCount) {
			*mNodesIndex = mConfiguration.mMaxNodeCount;
			return INVALID_NODE_INDEX;
		}

		mNodes[oldVal] = *mNodeForInitialization; // intialize
		return (int) oldVal;
	}

	KRR_DEVICE Node *getNodeByIndex(int idx) { return mNodes + idx; }

	// node won't update when depth >= maxDepth
	// now trainingNode can not be leaf
	KRR_DEVICE void lookupMaxDepth(Vector3f posWorld, const Vector3f dir,
								   const SamplingNode *&sampling, TrainingNode *&training,
								   Node *&trainingPosNode, const int maxDepth) {

		const uint bin	 = mapOutgoingDirectionToHistogramBin(dir);
		Vector3f pos	 = mapPointToUnitCube(posWorld);
		uint currentuint = 0;
		int depthAcc	 = 0;
		while (true) {
			int stratum		   = stratumIndex(pos);
			Node *node		   = mNodes + currentuint;
			Node::Child &child = node->mChildren[stratum];
			if (currentuint == 0 || child.mSampling[bin].isValid()) {
				/// a valid node for sampling
				sampling = &child.mSampling[bin];
			}

			training		= &child.mTraining[bin];
			trainingPosNode = node;
			if (child.isLeaf()) {
				/// reached a leaf node
				break;
			}

			if (++depthAcc > maxDepth) {
				break;
			}

			currentuint = child.mIndex;
		}
	}

	KRR_DEVICE void lookup(Vector3f posWorld, const Vector3f dir, const SamplingNode *&sampling,
						   TrainingNode *&training, Node *&trainingPosNode) {

		const uint bin	 = mapOutgoingDirectionToHistogramBin(dir);
		Vector3f pos	 = mapPointToUnitCube(posWorld);
		uint currentuint = 0;
		while (true) {
			int stratum		   = stratumIndex(pos);
			Node *node		   = mNodes + currentuint;
			Node::Child &child = node->mChildren[stratum];
			if (currentuint == 0 || child.mSampling[bin].isValid()) {
				/// a valid node for sampling
				sampling = &child.mSampling[bin];
			}

			if (child.isLeaf()) {
				/// reached a leaf node
				training		= &child.mTraining[bin];
				trainingPosNode = node;
				break;
			}

			currentuint = child.mIndex;
		}
	}

	// [note] you should call this when mNodes[] is exhausted, as new node have no data
	KRR_DEVICE void lookupIgnoreInvalid(Vector3f posWorld, const Vector3f dir,
										const SamplingNode *&sampling, TrainingNode *&training,
										Node *&trainingPosNode) {

		const uint bin	 = mapOutgoingDirectionToHistogramBin(dir);
		Vector3f pos	 = mapPointToUnitCube(posWorld);
		uint currentuint = 0;
		while (true) {
			int stratum		   = stratumIndex(pos);
			Node *node		   = mNodes + currentuint;
			Node::Child &child = node->mChildren[stratum];
			if (child.isLeaf()) {
				/// reached a leaf node
				sampling		= &child.mSampling[bin];
				training		= &child.mTraining[bin];
				trainingPosNode = node;
				break;
			}

			currentuint = child.mIndex;
		}
	}
};

NAMESPACE_END(krr)

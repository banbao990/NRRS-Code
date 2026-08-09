#pragma once
#include <atomic>
#include <array>
#include <stack>

#include "common.h"
#include "sampler.h"
#include "logger.h"
#include "device/gpustd.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/atomic.h"

#include "util/check.h"

NAMESPACE_BEGIN(krr)

using AtomicType = double; /* The type used for storing atomic data, e.g., per-node irradiance. */

enum class ESampleCombination {
	EDiscardWithAutomaticBudget,
	EWeightBySampleCount,
};

enum class EBsdfSamplingFractionLoss {
	ENone,
	EKL,
	EVariance,
};

enum class ESpatialFilter {
	ENearest,
	EStochasticBox,
	EBox,
};

enum class EDirectionalFilter {
	ENearest,
	EBox,
};

enum class EDistribution {
	ERadiance,
	ESimple,
	EFull
};

KRR_ENUM_DEFINE(ESampleCombination,
				{
					{ESampleCombination::EDiscardWithAutomaticBudget, "automatic"},
					{ESampleCombination::EWeightBySampleCount, "all"},
				});

KRR_ENUM_DEFINE(EDirectionalFilter, {
										{EDirectionalFilter::ENearest, "nearest"},
										{EDirectionalFilter::EBox, "box"},
									});

KRR_ENUM_DEFINE(ESpatialFilter, {
									{ESpatialFilter::ENearest, "nearest"},
									{ESpatialFilter::EStochasticBox, "stochastic"},
									{ESpatialFilter::EBox, "box"},
								});

KRR_ENUM_DEFINE(EDistribution, {
								   {EDistribution::ERadiance, "radiance"},
								   {EDistribution::ESimple, "simple"},
								   {EDistribution::EFull, "full"},
							   });

template <typename T> class DTree; // for QuadTreeNode

template <typename T> class QuadTreeNode {
public:
	QuadTreeNode() = default;

	KRR_HOST void initialize();

	KRR_HOST void setSum(int index, T val) { mSum[index].store(val); }

	KRR_HOST void resetZero(int index) { mSum[index].store(0.0); }

	KRR_CALLABLE float sum(int index) const { return mSum[index].load(); }

	KRR_CALLABLE T sumAll(int index) const { return mSum[index].loadAll(); }

	KRR_HOST void copyFrom(const QuadTreeNode &arg);

	KRR_HOST QuadTreeNode(const QuadTreeNode &arg);

	KRR_HOST QuadTreeNode &operator=(const QuadTreeNode &arg);

	KRR_CALLABLE void setChild(int idx, uint16_t val) { mChildren[idx] = val; }

	KRR_CALLABLE uint16_t child(int idx) const { return mChildren[idx]; }

	KRR_HOST void setSum(T val) {
		for (int i = 0; i < 4; ++i) {
			setSum(i, val);
		}
	}

	KRR_HOST void resetZero() {
		for (int i = 0; i < 4; ++i) {
			resetZero(i);
		}
	}

	KRR_CALLABLE int childIndex(Vector2f &p) const {
		int res = 0;
		for (int i = 0; i < Vector2f::dim; ++i) {
			if (p[i] < 0.5f) {
				p[i] *= 2;
			} else {
				p[i] = (p[i] - 0.5f) * 2;
				res |= 1 << i;
			}
		}

		return res;
	}

	// Evaluates the directional irradiance *sum density* (i.e. sum / area) at a given location p.
	// To obtain radiance, the sum density (result of this function) must be divided
	// by the total statistical weight of the estimates that were summed up.
	KRR_CALLABLE float eval(Vector2f &p, const TypedBuffer<QuadTreeNode> &nodes) const {
		CHECK(p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
		const int index = childIndex(p);
		if (isLeaf(index)) {
			return 4 * sum(index);
		} else {
			return 4 * nodes[child(index)].eval(p, nodes);
		}
	}

	KRR_CALLABLE float pdf(Vector2f &p, const TypedBuffer<QuadTreeNode> &nodes) const {
		CHECK(p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
		const int index = childIndex(p);
		if (!(sum(index) > 0)) {
			return 0;
		}

		const float factor = 4 * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
		if (isLeaf(index)) {
			return factor;
		} else {
			return factor * nodes[child(index)].pdf(p, nodes);
		}
	}

	KRR_CALLABLE int depthAt(Vector2f &p, const TypedBuffer<QuadTreeNode> &nodes) const {
		CHECK(p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
		const int index = childIndex(p);
		if (isLeaf(index)) {
			return 1;
		} else {
			return 1 + nodes[child(index)].depthAt(p, nodes);
		}
	}

	/*	Sample a point according to the sum radiance of the 4 children nodes...
	 *	*---*---*	0 —— x —— 1
	 *  | 0 | 1 |	|
	 *  *---*---*	y
	 *  | 2 | 3 |	|
	 *  *---*---*	1
	 */
	KRR_CALLABLE Vector2f sample(Sampler &sampler, const TypedBuffer<QuadTreeNode> &nodes) const {
		int index = 0;

		float topLeft  = sum(0);
		float topRight = sum(1);
		float partial  = topLeft + sum(2);
		float total	   = partial + topRight + sum(3);

		// Should only happen when there are numerical instabilities.
		if (!(total > 0.0f)) {
			return sampler.get2D();
		}

		float boundary	= partial / total;
		Vector2f origin = Vector2f{0.0f, 0.0f}; /* x, y */

		float sample = sampler.get1D();

		if (sample < boundary) { /* whether sampled the left part (left -> x < 0.5) */
			CHECK(partial > 0);
			sample /= boundary;			  /* sample reuse */
			boundary = topLeft / partial; /* next check whether sampled the top left part */
		} else {
			partial = total - partial; /* no, in the right part... */
			CHECK(partial > 0);
			origin[0] = 0.5f; /* move to the right part */
			sample	  = (sample - boundary) / (1.0f - boundary);
			boundary  = topRight / partial;
			index |= 1 << 0; /* the cell #1 or #3... */
		}

		if (sample < boundary) {
			sample /= boundary;
		} else {
			origin[1] = 0.5f;
			sample	  = (sample - boundary) / (1.0f - boundary);
			index |= 1 << 1;
		}

		if (isLeaf(index)) {
			return origin + 0.5f * sampler.get2D();
		} else {
			return origin + 0.5f * nodes[child(index)].sample(sampler, nodes);
		}
	}

	KRR_DEVICE void record(Vector2f &p, T irradiance, TypedBuffer<QuadTreeNode> &nodes) {
		CHECK(p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1);
		int index = childIndex(p);

		if (isLeaf(index)) {
			mSum[index].add(irradiance);
		} else {
			nodes[child(index)].record(p, irradiance, nodes);
		}
	}

	KRR_CALLABLE float computeOverlappingArea(const Vector2f &min1, const Vector2f &max1,
											  const Vector2f &min2, const Vector2f &max2) {
		float lengths[2];
		for (int i = 0; i < 2; ++i) {
			lengths[i] = max(min(max1[i], max2[i]) - max(min1[i], min2[i]), 0.0f);
		}
		return lengths[0] * lengths[1];
	}

	/* For performance consideration, this routine is executed parallely on device. */
	/* This RECORD splatts the contribution of an radiance record to nearby cells. */
	KRR_DEVICE void record(const Vector2f &origin, float size, Vector2f nodeOrigin, float nodeSize,
						   const T value, TypedBuffer<QuadTreeNode> &nodes) {
		float childSize = nodeSize / 2;
		for (int i = 0; i < 4; ++i) {
			Vector2f childOrigin = nodeOrigin;
			if (i & 1) {
				childOrigin[0] += childSize;
			}
			if (i & 2) {
				childOrigin[1] += childSize;
			}

			float w = computeOverlappingArea(origin, origin + Vector2f(size), childOrigin,
											 childOrigin + Vector2f(childSize));
			if (w > 0.0f) {
				if (isLeaf(i)) {
					mSum[i].add(value * w);
				} else {
					nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes);
				}
			}
		}
	}

	KRR_CALLABLE bool isLeaf(int index) const { return child(index) == 0; }

	/*	Ensure that each quadtree node's sum of irradiance estimates equals that of all its
	   children. This function do not change the topology of the D-Tree. */
	KRR_HOST void build(std::vector<QuadTreeNode> &nodes,
						EDistribution distribution = EDistribution::ERadiance,
						float parentSize		   = 1.f);

private:
	friend class DTree<T>;

	T mSum[4]{0};
	uint16_t mChildren[4]{};
};

template <typename T> class DTree {
public:
	DTree() = default;

	KRR_HOST void initialize();

	KRR_HOST void clear();

	KRR_HOST DTree &operator=(const DTree &other);

	KRR_HOST DTree(const DTree &other);

	KRR_CALLABLE QuadTreeNode<T> &node(size_t i) { return mNodes[i]; }

	KRR_CALLABLE const QuadTreeNode<T> &node(size_t i) const { return mNodes[i]; }

	KRR_CALLABLE float mean() const {
		float statisticalWeight = mStatisticalWeight.load();
		if (statisticalWeight == 0) {
			return 0;
		}
		const float factor = 1 / (M_4PI * statisticalWeight);
		return factor * mSum.load();
	}

	KRR_CALLABLE T meanAll() const {
		float statisticalWeight = mStatisticalWeight.load();
		if (statisticalWeight == 0) {
			return 0;
		}
		const float factor = 1 / (M_4PI * statisticalWeight);
		return mSum.loadAll() * factor;
	}

	/* Irradiance is radiance/wiPdf, statistical weight is generally a constant value. */
	KRR_DEVICE void recordIrradiance(Vector2f p, T irradiance, float statisticalWeight,
									 EDirectionalFilter directionalFilter) {
		if (!isinf(statisticalWeight) && statisticalWeight > 0) {
			mStatisticalWeight.fetch_add((AtomicType) statisticalWeight);

			if (!irradiance.isinf() && irradiance.load() > 0) {
				if (directionalFilter == EDirectionalFilter::ENearest) {
					mNodes[0].record(p, irradiance * statisticalWeight, mNodes);
				} else {
					int depth  = depthAt(p);
					float size = pow(0.5f, depth);

					Vector2f origin = p;
					origin[0] -= size / 2;
					origin[1] -= size / 2;
					mNodes[0].record(origin, size, Vector2f(0.0f), 1.0f,
									 irradiance * statisticalWeight / (size * size), mNodes);
				}
			}
		}
	}

	KRR_CALLABLE float pdf(Vector2f p) const {
		if (!(mean() > 0)) {
			return M_INV_4PI;
		}
		return mNodes[0].pdf(p, mNodes) * M_INV_4PI;
	}

	KRR_CALLABLE int depthAt(Vector2f p) const { return mNodes[0].depthAt(p, mNodes); }

	KRR_CALLABLE int depth() const { return mMaxDepth; }

	KRR_CALLABLE Vector2f sample(Sampler &sampler) const {
		if (!(mean() > 0)) { /* This d-tree has no radiance records. */
			return sampler.get2D();
		}
		Vector2f res = mNodes[0].sample(sampler, mNodes);
		return clamp(res, 0.f, 1.f);
	}

	KRR_CALLABLE size_t numNodes() const { return mNodes.size(); }

	KRR_CALLABLE float statisticalWeight() const { return mStatisticalWeight.load(); }

	KRR_HOST void setStatisticalWeight(float statisticalWeight) {
		mStatisticalWeight.store((AtomicType) statisticalWeight);
	}

	/* This function adaptively subdivides / prunes the D-Tree, recursively in a sequential manner.
	 */
	KRR_HOST void reset(const DTree &previousDTree, int newMaxDepth, float subdivisionThreshold);

	KRR_CALLABLE size_t approxMemoryFootprint() const {
		return mNodes.sizeInBytes() * sizeof(QuadTreeNode<T>) + sizeof(*this);
	}

	KRR_HOST void build(EDistribution distribution = EDistribution::ERadiance);

private:
	atomic<AtomicType> mStatisticalWeight{0};
	T mSum{0};

	int mMaxDepth{0};
	/* These data resides the device memory, on the device-side S-Tree. */
	TypedBuffer<QuadTreeNode<T>> mNodes{};
};

template <typename T> struct DTreeRecord {
	Vector3f mDir;
	T mRadiance;
	float mWiPdf, mBsdfPdf, mDTreePdf;
	float mStatisticalWeight;
	bool mIsDelta;
};

template <typename T> struct DTreeWrapper {
public:
	DTreeWrapper() = default;

	KRR_HOST void initialize();

	KRR_HOST void clear();

	KRR_HOST DTreeWrapper &operator=(const DTreeWrapper &other);

	KRR_DEVICE void record(const DTreeRecord<T> &rec, EDirectionalFilter directionalFilter,
						   EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
		if (!rec.mIsDelta) {
			T irradiance = rec.mRadiance / rec.mWiPdf;
			mBuilding.recordIrradiance(dirToCanonical(rec.mDir), irradiance, rec.mStatisticalWeight,
									   directionalFilter);
		}
	}

	/* This uniformly maps the 2D area to the unit spherical surface. */
	KRR_CALLABLE static Vector3f canonicalToDir(Vector2f p) {
		const float cosTheta = 2 * p[0] - 1;
		const float phi		 = M_2PI * p[1];

		const float sinTheta = safe_sqrt(1 - cosTheta * cosTheta);
		float sinPhi = sin(phi), cosPhi = cos(phi);

		return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
	}

	KRR_CALLABLE static Vector2f dirToCanonical(const Vector3f &d) {
		if (d.hasInf()) return {};

		const float cosTheta = clamp(d[2], -1.f, 1.f);
		float phi			 = std::atan2(d[1], d[0]);
		while (phi < 0) phi += M_2PI;

		return {(cosTheta + 1) / 2, phi / M_2PI};
	}

	KRR_HOST void build(EDistribution distribution);

	KRR_HOST void reset(int maxDepth, float subdivisionThreshold);

	KRR_CALLABLE Vector3f sample(Sampler &sampler) const {
		return canonicalToDir(mSampling.sample(sampler));
	}

	KRR_CALLABLE float pdf(const Vector3f &dir) const { return mSampling.pdf(dirToCanonical(dir)); }

	KRR_CALLABLE float diff(const DTreeWrapper &other) const { return 0.0f; }

	KRR_CALLABLE int depth() const { return mSampling.depth(); }

	KRR_CALLABLE size_t numNodes() const { return mSampling.numNodes(); }

	KRR_CALLABLE float meanRadiance() const { return mSampling.mean(); }

	KRR_CALLABLE T meanRadianceAll() const { return mSampling.meanAll(); }

	KRR_CALLABLE float statisticalWeight() const { return mSampling.statisticalWeight(); }

	KRR_CALLABLE float statisticalWeightBuilding() const { return mBuilding.statisticalWeight(); }

	KRR_HOST void setStatisticalWeightBuilding(float statisticalWeight) {
		mBuilding.setStatisticalWeight(statisticalWeight);
	}

	KRR_CALLABLE size_t approxMemoryFootprint() const {
		return mBuilding.approxMemoryFootprint() + mSampling.approxMemoryFootprint();
	}

private:
	DTree<T> mBuilding;
	DTree<T> mSampling;
};

template <typename T> struct STreeNode {
	STreeNode() = default;

	KRR_HOST void initialize();

	KRR_CALLABLE int childIndex(Vector3f &p) const {
		if (p[mAxis] < 0.5f) {
			p[mAxis] *= 2;
			return 0;
		} else {
			p[mAxis] = (p[mAxis] - 0.5f) * 2;
			return 1;
		}
	}

	KRR_CALLABLE int nodeIndex(Vector3f &p) const { return mChildren[childIndex(p)]; }

	KRR_CALLABLE DTreeWrapper<T> *dTreeWrapper(Vector3f &p, Vector3f &size,
											   TypedBuffer<STreeNode> &nodes) {
		CHECK(p[mAxis] >= 0 && p[mAxis] <= 1);
		if (mIsLeaf) {
			return &mDTree;
		} else {
			size[mAxis] /= 2;
			return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes);
		}
	}

	KRR_CALLABLE const DTreeWrapper<T> *dTreeWrapper() const { return &mDTree; }

	KRR_CALLABLE int depth(Vector3f &p, const TypedBuffer<STreeNode> &nodes) const {
		CHECK(p[mAxis] >= 0 && p[mAxis] <= 1);
		if (mIsLeaf) {
			return 1;
		} else {
			return 1 + nodes[nodeIndex(p)].depth(p, nodes);
		}
	}

	KRR_CALLABLE int depth(const TypedBuffer<STreeNode> &nodes) const {
		int result = 1;

		if (!mIsLeaf) {
			for (auto c : mChildren) {
				result = max(result, 1 + nodes[c].depth(nodes));
			}
		}

		return result;
	}

	KRR_HOST void forEachLeaf(
		std::function<void(const DTreeWrapper<T> *, const Vector3f &, const Vector3f &)> func,
		Vector3f p, Vector3f size, const TypedBuffer<STreeNode<T>> &nodes) const;

	KRR_CALLABLE float computeOverlappingVolume(const Vector3f &min1, const Vector3f &max1,
												const Vector3f &min2, const Vector3f &max2) {
		float lengths[3];
		for (int i = 0; i < 3; ++i) {
			lengths[i] = max(min(max1[i], max2[i]) - max(min1[i], min2[i]), 0.0f);
		}
		return lengths[0] * lengths[1] * lengths[2];
	}

	KRR_DEVICE void record(const Vector3f &min1, const Vector3f &max1, Vector3f min2,
						   Vector3f size2, const DTreeRecord<T> &rec,
						   EDirectionalFilter directionalFilter,
						   EBsdfSamplingFractionLoss bsdfSamplingFractionLoss,
						   TypedBuffer<STreeNode> &nodes) {

		float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
		if (w > 0) {
			if (mIsLeaf) {
				mDTree.record({rec.mDir, rec.mRadiance, rec.mWiPdf, rec.mBsdfPdf, rec.mDTreePdf,
							   rec.mStatisticalWeight * w, rec.mIsDelta},
							  directionalFilter, bsdfSamplingFractionLoss);
			} else {
				size2[mAxis] /= 2;
				for (int i = 0; i < 2; ++i) {
					if (i & 1) {
						min2[mAxis] += size2[mAxis];
					}

					nodes[mChildren[i]].record(min1, max1, min2, size2, rec, directionalFilter,
											   bsdfSamplingFractionLoss, nodes);
				}
			}
		}
	}

	bool mIsLeaf{true};
	DTreeWrapper<T> mDTree;
	int mAxis{0};
	uint32_t mChildren[2];
};

template <typename T> class STree {
public:
	STree() = default;

	KRR_HOST STree(const AABB &aabb, Allocator alloc);

	KRR_HOST void clear();

	KRR_HOST void subdivideAll();

	/* This is the actual function that directly changes the topology of the S-Tree. */
	KRR_HOST void subdivide(int nodeIdx, std::vector<STreeNode<T>> &nodes);

	/* This function also returns the size of the voxel where p resides. */
	KRR_CALLABLE DTreeWrapper<T> *dTreeWrapper(Vector3f p, Vector3f &size) {
		size = mAabb.diagonal();
		p	 = p - mAabb.min();
		p	 = p.cwiseQuotient(size);
		return mNodes[0].dTreeWrapper(p, size, mNodes);
	}

	KRR_CALLABLE DTreeWrapper<T> *dTreeWrapper(Vector3f p) {
		Vector3f size;
		return dTreeWrapper(p, size);
	}

	KRR_HOST void forEachDTreeWrapper(std::function<void(DTreeWrapper<T> *)> func);

	KRR_DEVICE void record(const Vector3f &p, const Vector3f &dTreeVoxelSize, DTreeRecord<T> rec,
						   EDirectionalFilter directionalFilter,
						   EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
		float volume = 1;
		for (int i = 0; i < 3; ++i) {
			volume *= dTreeVoxelSize[i];
		}
		rec.mStatisticalWeight /= volume;
		mNodes[0].record(p - dTreeVoxelSize * 0.5f, p + dTreeVoxelSize * 0.5f, mAabb.min(),
						 mAabb.diagonal(), rec, directionalFilter, bsdfSamplingFractionLoss,
						 mNodes);
	}

	KRR_HOST static bool STree::shallSplit(const STreeNode<T> &node, size_t samplesRequired);

	/*	Only this function would change the topology of the S-Tree, sequentially executed.
		To adaptively subdivision, should be run on hostcode? */
	KRR_HOST void refine(size_t sTreeThreshold, int maxMB);

	KRR_HOST void gatherStatistics() const;

	KRR_CALLABLE const AABB &aabb() const { return mAabb; }

private:
	TypedBuffer<STreeNode<T>> mNodes;
	AABB mAabb;
};

NAMESPACE_END(krr)
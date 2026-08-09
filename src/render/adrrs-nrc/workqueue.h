#pragma once
#include <atomic>

#include "common.h"
#include "device/cuda.h"
#include "logger.h"
#include "util/math_utils.h"

#include "workitem.h"
#include "render/profiler/profiler.h"
#include "render/nrc/workqueue.h"
#include "render/common/commonworkqueue.h"

NAMESPACE_BEGIN(krr)

class ADNInferenceQueue : public WorkQueue<ADNInferenceWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(ADNInferenceQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((byte *) &(this->m_size)) - ((byte *) this));
			Log(Info, "[ADNInferenceQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

NAMESPACE_END(krr)
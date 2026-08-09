#pragma once

#include "render/wavefront/workqueue.h"
#include "commonitem.h"

NAMESPACE_BEGIN(krr)

class TidQueue : public WorkQueue<TidWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(uint tid) {
		int last		 = m_size.fetch_add(1);
		this->mTid[last] = tid;
		return last;
	}

	KRR_CALLABLE int push(uint tid, const int cnt) {
		int last = m_size.fetch_add(cnt);
		for (int i = 0; i < cnt; ++i) {
			this->mTid[last + i] = tid;
		}
		return last;
	}

	KRR_CALLABLE void pushCheck(uint tid) {
		int last = m_size.fetch_add(1);
		if (last < nAlloc) {
			this->mTid[last] = tid;
		}
	}

	KRR_CALLABLE void pushCheck(uint tid, const int cnt) {
		int last = m_size.fetch_add(cnt);
		for (int i = 0; i < cnt; ++i) {
			const int j = last + i;
			if (j < nAlloc) {
				this->mTid[j] = tid;
			}
		}
	}

	inline uint offsetOfSize() {
		// TODO: offsetof applied to a type other than a standard-layout class type
		// constexpr int off = offsetof(TidQueue, m_size);

		static int offset = -1;
		if (offset == -1) {
			offset = sizeof(char) * (((byte *) &(this->m_size)) - ((byte *) this));
			Log(Info, "[TidQueue] offset of size: %d", offset);
		}
		return offset;
	}
};

NAMESPACE_END(krr)

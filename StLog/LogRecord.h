#pragma once
#include <chrono>
#include <unordered_map>
#include <atomic>
#include <assert.h>
#include <mutex>
#include <Windows.h>

enum class Severity : uint16_t
{
	Trace,
	Debug,
	Info,
	Warn,
	Error,
	Fatal,
};

template<typename T>
struct Span
{
	Span()
	{
		data = 0;
		len = 0;
	}

	Span(const char* d)
	{
		data = d;
		len = strlen(d);
	}

	Span(T* d, size_t l)
	{
		data = d;
		len = l;
	}

	//! Implicit cast to std::string
	operator std::string() const
	{
		return std::string(data, len);
	}

	bool operator==(const Span& rhs) const
	{
		if (len == rhs.len)
		{
			return memcmp(rhs.data, data, this->len)==0;
		}
		return false;
	}

	T* data;
	size_t len;
};

namespace std
{
	template<>
	struct hash<Span<const char>>
	{
		std::size_t operator()(const Span<const char>& k) const
		{
			std::size_t result = 2166136261;
			std::size_t PRIME = 16777619;

			// simple FNV hash
			for (size_t i = 0; i < k.len; i++)
			{
				char next = k.data[i];
				result = (result ^ next) * PRIME;
			}
			return (std::size_t)k.data;
		}
	};
}
enum class AttributeType : uint8_t
{
	type_u8,
	type_u16,
	type_u32,
	type_u64,
	type_s8,
	type_s16,
	type_s32,
	type_s64,
	type_bool,
	type_f32,
	type_f64,
	type_string_view,   // pointer to existing memory
};

class AttributeValue
{
public:
	AttributeType  data_type;
	union U
	{
		uint8_t     u8;
		uint16_t    u16;
		uint32_t    u32;
		uint64_t    u64;
		int8_t      s8;
		int16_t     s16;
		int32_t     s32;
		int64_t     s64;
		bool        b;
		float       f32;
		double      f64;
		Span<const char> s;	

		U() 
		{ 
		   u8 = 0; 
		}
		~U() {}
	}  data;

	AttributeValue()
	{
		data_type = AttributeType::type_u8;
	}

	AttributeValue(uint8_t val)
	{
		data.u8 = val;
		data_type = AttributeType::type_u8;
	}

	AttributeValue(uint16_t val)
	{
		data.u16 = val;
		data_type = AttributeType::type_u16;
	}

	AttributeValue(uint32_t val)
	{
		data.u32 = val;
		data_type = AttributeType::type_u32;
	}

	AttributeValue(uint64_t val)
	{
		data.u64 = val;
		data_type = AttributeType::type_u64;
	}

	AttributeValue(unsigned long val)
	{
		data.u64 = val;
		data_type = (sizeof(val) == 8) 
			      ? AttributeType::type_u64
			      : AttributeType::type_u32
			      ;
	}
	AttributeValue(long val)
	{
		data.u64 = val;
		data_type = (sizeof(val) == 8)
			? AttributeType::type_s64
			: AttributeType::type_s32
			;
	}

	AttributeValue(int8_t val)
	{
		data.s8 = val;
		data_type = AttributeType::type_s8;
	}

	AttributeValue(int16_t val)
	{
		data.s16 = val;
		data_type = AttributeType::type_s16;
	}

	AttributeValue(int32_t val)
	{
		data.s32 = val;
		data_type = AttributeType::type_s32;
	}

	AttributeValue(int64_t val)
	{
		data.s64 = val;
		data_type = AttributeType::type_s64;
	}

	AttributeValue(bool val)
	{
		data.b = val;
		data_type = AttributeType::type_bool;
	}

	AttributeValue(float val)
	{
		data.f32 = val;
		data_type = AttributeType::type_f32;
	}

	AttributeValue(double val)
	{
		data.f64 = val;
		data_type = AttributeType::type_f64;
	}

	// Only holds pointer to string
	// to by copied in Exporter
	AttributeValue(const char* const str)
	{
		data.s = str;
		data_type = AttributeType::type_string_view;
	}
	AttributeValue(const std::string& str)
	{
		data.s.data = str.c_str();
		data.s.len = str.length();
		data_type = AttributeType::type_string_view;
	}
private:
};

template<typename Visitor, typename R=void>
R visit(Visitor&& v, const AttributeValue& val)
{
	switch (val.data_type)
	{
	case AttributeType::type_bool: return v(val.data.b); break;
	case AttributeType::type_u8:   return v(val.data.u8); break;
	case AttributeType::type_s8:   return v(val.data.s8); break;
	case AttributeType::type_u16:  return v(val.data.u16); break;
	case AttributeType::type_s16:  return v(val.data.s16); break;
	case AttributeType::type_f32:  return v(val.data.f32); break;
	case AttributeType::type_u32:  return v(val.data.u32); break;
	case AttributeType::type_s32:  return v(val.data.s32); break;
	case AttributeType::type_f64:  return v(val.data.f64); break;
	case AttributeType::type_u64:  return v(val.data.u64); break;
	case AttributeType::type_s64:  return v(val.data.s64); break;
	case AttributeType::type_string_view: return v(val.data.s); break;
	}
}

struct NamedAttribute
{
	const char* name;
	AttributeValue value;
};


class AttributeList
{
public:
	virtual ~AttributeList() {}
	virtual void add_attribute(const NamedAttribute& attr) = 0;
	virtual NamedAttribute* begin() = 0;
	virtual NamedAttribute* end() = 0;
};

/**
 A record of a log entry
 */
class LogRecord
{
public:
	virtual void set_timestamp(uint64_t nsec) = 0;
	virtual void set_severity(Severity severity) = 0;
	void set_name(const char* name)
	{
		set_name( {name, strlen(name)} );
	}
	void set_message(const char* name)
	{
		set_message({ name, strlen(name) });
	}

	virtual void set_name(Span<const char>&& name) = 0;
	virtual void set_message(Span<const char>&& message) = 0;
	virtual void add_attribute(const Span<const char>& key, const AttributeValue& attr) = 0;
};

/**
  Receives a LogRecords and forwards to a LogExportor.
  Allowed to buffer records before forwarding
 */
class LogProcessor
{
public:
	//! Creates new LogRecord to be filled by Logger
	virtual LogRecord* create_record() = 0;

	virtual AttributeList* create_attribute_list() = 0;
	/**
	Adds a record to be processed.  Processor will
	release resources for the record
	*/
	virtual void add_record(LogRecord* rec) = 0;

};

class LogExporter
{
public:
	//! Creates a new LogRecord that can be exported
	virtual LogRecord* create_record() = 0;

	virtual AttributeList* create_attribute_list() = 0;

	//! Export a record and release resources 
	virtual void export_record(LogRecord* rec) = 0;
};


class Logger;

class LogScope
{
	friend Logger;
	AttributeList* attrs;
	LogScope* next;
	Logger* logger;

	// called when LogScope is created in Logger
	LogScope(AttributeList* at)
	{
		logger = 0;
		next = 0;
		attrs = at;
	}
public:
	// called when LogScope is returned from Logger
	LogScope(LogScope&& other);

	~LogScope()
	{
		delete attrs;
		// implicit end of scope
		end();
	}

	// explicit end of scope
	void end();
};

class Logger
{
public:
	Logger(const char* name, LogProcessor& proc)
	: m_processor(proc)
	{
		m_name = name;
	}

	/**
	 Begin a Scope block to attach extra attributes
	 */
	LogScope begin_scope(std::initializer_list<NamedAttribute> at)
	{
		AttributeList* attrs = m_processor.create_attribute_list();

		for (const NamedAttribute* it = at.begin();
			it < at.end();
			it++)
		{
			attrs->add_attribute(*it);
		}
		LogScope scope(attrs);
		scope.logger = this;
		scope.next = m_scope;
		m_scope = &scope;
		return scope;
	}

	void log(Severity sev, const char* msg, std::initializer_list<NamedAttribute> attrs)
	{
		LogRecord* r = m_processor.create_record();
		uint64_t nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::system_clock::now().time_since_epoch()
			).count();

		r->set_timestamp(nsec);
		r->set_name(m_name);
		r->set_message(msg);
		r->set_severity(sev);

		const NamedAttribute* it;

		// Add scope values
		LogScope* curr = m_scope;
		int depth = 0;
		while (curr)
		{
			for (it = curr->attrs->begin(); it < curr->attrs->end(); it++)
			{
				r->add_attribute(it->name, it->value);
			}
			curr = curr->next;
		}

		// add local values
		for (it = attrs.begin(); it < attrs.end(); it++)
		{
			r->add_attribute(it->name, it->value);
		}

		m_processor.add_record(r);
	}

private:
	friend LogScope;

	LogProcessor& m_processor;
	const char*   m_name;

	static thread_local LogScope* m_scope;
};

inline void LogScope::end()
{
	if (logger == 0) return;
	logger->m_scope = next;

	logger = 0;
}

LogScope::LogScope(LogScope&& other)
{
	logger = std::move(other.logger);
	attrs  = std::move(other.attrs);
	next   = std::move(other.next);

	assert(logger->m_scope == &other);
	logger->m_scope = this;

	other.logger = nullptr;
	other.next = nullptr;
	other.attrs = nullptr;
}

class SimpleLogProcessor : public LogProcessor
{
public:
	SimpleLogProcessor(LogExporter&& e)
	: m_exporter( e )
	{

	}

	virtual AttributeList* create_attribute_list()
	{
		return m_exporter.create_attribute_list();
	}
	//! Creates new LogRecord to be filled by Logger
	virtual LogRecord* create_record()
	{
		return m_exporter.create_record();
	}
	/**
	Adds a record to be processed.  Processor will
	release resources for the record
	*/
	virtual void add_record(LogRecord* rec)
	{
		// no buffering, pass directly on
		m_exporter.export_record(rec);
	}
private:

	LogExporter& m_exporter;
};

struct AllocBackEnd
{
	static const int COUNT = 100;
	static const int BUFFERS = 2;
	char* buff[4];

	// to help in lock-free need to know when index
	// has changed and been set back.  To do this there
	// is a sequence modification counter.  This limits
	// to only 64K items in the list, but this should
	// plenty large for what we need here.
	struct SeqCounter
	{
		uint16_t index;
		uint16_t seq;
	};
	std::atomic<SeqCounter> free_list[4];
	size_t size[4];

	std::atomic<int> logi;
	std::atomic<int> FLUSH;
	struct L{
		uint64_t start;
		uint64_t ts;
		DWORD    tid;
		size_t   size;
		uint8_t  curr;
		uint8_t  next;
		uint8_t  alloc;
	};
	volatile L log[2048];

	void push(int curr, int next, uint8_t alloc, int size, uint64_t start)
	{
		int ind;
		do
		{
			ind = logi;
		} while (!logi.compare_exchange_strong(ind, (ind + 1) & 2047));
		log[ind].start = start;
		log[ind].ts = std::chrono::system_clock::now().time_since_epoch().count();
		log[ind].tid = GetCurrentThreadId();
		log[ind].size = size;
		log[ind].curr = curr;
		log[ind].next = next;
		log[ind].alloc = alloc;
	}

	AllocBackEnd()
	{
		FLUSH = 0;
		logi = 0;

		size[0] = 64;
		size[1] = 128;
		size[2] = 256;
		size[3] = 1024;


		for (int i = 0; i < BUFFERS; i++)
		{
			free_list[i] = {0,0};
		}

		for (int k = 0; k < BUFFERS; k++)
		{
			buff[k] = (char*)malloc(size[k] * COUNT);
			int* p = (int*)buff[k];
			assert(p != 0);
			for (int i = 0; i < COUNT; i++)
			{
				p[i * size[k] / sizeof(int)] = i + 1;
			}

			// last pointer is invalid
			p[(COUNT-1) * size[k] / sizeof(int)] = -1;
		}
	}
	void DO_FLUSH()
	{
		int exp = 0;
		while(!FLUSH.compare_exchange_strong(exp,1))
		{
			Sleep(1000);
		}

		FLUSH = 1;

		FILE* f;
		fopen_s(&f, "trace.log", "w");

		for (int i = 0; i < 2048; i++)
		{
			volatile L& l = log[i];
			fprintf(f, "%llu %llu %d %d %d %d %d\n", 
				l.start,
				l.ts,
				l.tid,
				l.size,
				l.curr,
				l.next,
				l.alloc
				);
		}
		fclose(f);
	}
	void* alloc(size_t n)
	{
		while (FLUSH) Sleep(1000);

		void* rst = 0;

		// find the smallest size that meets the requested size
		for (int i = 0; i < BUFFERS; i++)
		{
			if (n > size[i]) continue;

			int* next = 0;

			SeqCounter nxt;
			do
			{
				rst = 0;  // result pointer

				// latest free list pointer
				SeqCounter curr = free_list[i];

				if (curr.index == 0xFFFF)
					break;  // empty free list

				next  = (int*)&(buff[i][size[i] * (curr.index)]);

				// another thread pulls item out of free list
				// and inserts it again here..  *next would
				// now be wrong, as the buffer has changed

				nxt.index = *next;
				nxt.seq = curr.seq + 1;

				// if free_list changed, loop again
				if (!free_list[i].compare_exchange_weak(curr, nxt)) continue;

				break;
			}while(true);

			rst = next;
			break;
		}

		assert(rst != 0);
		return rst;
	}

	void dealloc(char* ptr)
	{
		while (FLUSH) Sleep(1000);
		for (int i = 0; i < BUFFERS; i++)
		{
			char* begin = buff[i];
			char* end = begin + COUNT * size[i];  // end of allocated range

			// in allocated memory range?
			if (ptr >= begin && ptr < end)
			{
				SeqCounter idx;
				idx.index = (ptr - begin) / size[i];  // block number;
				
				SeqCounter next;
				do
				{
					// get latest free_list pointer
					next = free_list[i];

					if (next.index == 0xFFFF)
					{
						break;
					}
					// safe to update ptr, as no other thread
					// should be accessing it right now
					*(int*)ptr = next.index;

					idx.seq = next.seq + 1;
				} while (!free_list[i].compare_exchange_weak(next, idx));
				//printf("Free %d=>%d\n", next, idx);
			
				break;
			}
		}
	}
};

/**
  Uses preallocated buffers
 */
template<typename T>
struct Alloc
{
	AllocBackEnd& be;
	typedef T value_type;
	typedef size_t size_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;

	template <class U> struct rebind { typedef Alloc<U> other; };

	Alloc(AllocBackEnd& b) : be(b)
	{
	}

	template<typename U>
	Alloc(Alloc<U> const& other) throw()
		: be(other.be)
	{
		int u = sizeof(U);
		int t = sizeof(T);
		// copy internal state
	}


	value_type* allocate(std::size_t n)
	{
		return (value_type*)be.alloc(n * sizeof(value_type));
	}

	void deallocate(value_type* p, std::size_t n) noexcept
	{
		be.dealloc((char*)p);
	}
};

/**
 Implements a LogRecord using a unordered_map with std::string for keys.
 This allows easiy construction and reading, but allocates memory as
 its built
 */
class SimpleLogRecord : public LogRecord
{
public:
	SimpleLogRecord()
	: m_attrs(16) // allocate space for 16 entries
	{
		ts = 0;
		severity = Severity::Trace;
		name.len = 0;
		message.len = 0;
	}

	virtual void set_timestamp(uint64_t nsec)
	{
		ts = nsec;
	}
	virtual void set_severity(Severity severity)
	{
		this->severity = severity;
	}

	// bring in the overloaded names
	using LogRecord::set_message;
	using LogRecord::set_name;

	virtual void set_name(Span<const char>&& name)
	{
		this->name = name;
	}
	virtual void set_message(Span<const char>&& message)
	{
		this->message = message;
	}

	std::list<std::string> m_strings;
	virtual void add_attribute(const Span<const char>& key, const AttributeValue& attr)
	{
		this->m_attrs.emplace(key, attr);
	}

	uint64_t    ts;
	Severity    severity;
	Span<const char> name;
	Span<const char> message;

	typedef std::string key_type;

	std::unordered_map<key_type, AttributeValue> m_attrs;
};

class NoAllocAttributeList : public AttributeList
{
public:
	std::vector<NamedAttribute, Alloc<NamedAttribute>> m_attrs;

	NoAllocAttributeList(AllocBackEnd& be)
	: m_attrs(Alloc<NamedAttribute>(be))
	{
	}

	void add_attribute(const NamedAttribute& attr)
	{
		m_attrs.push_back(attr);
	}

	NamedAttribute* begin()
	{
		return &m_attrs[0];
	}

	NamedAttribute* end()
	{
		return begin() + m_attrs.size();
	}
};

/**
  Does not allocate memory.  Uses AllocBackEnd to get
  memory it needs.   Allows for quick building of record and in places
  that may not allow memory to be allocated
 */
class NoMallocLogRecord : public LogRecord
{
public:
	NoMallocLogRecord(AllocBackEnd& be)
		: m_attrs(Alloc< decltype(m_attrs)::value_type>(be))
	{
		ts = 0;
		severity = Severity::Trace;
		name.len = 0;
		message.len = 0;
	}

	virtual void set_timestamp(uint64_t nsec)
	{
		ts = nsec;
	}
	virtual void set_severity(Severity severity)
	{
		this->severity = severity;
	}

	// bring in the overloaded names
	using LogRecord::set_message;
	using LogRecord::set_name;

	virtual void set_name(Span<const char>&& name)
	{
		this->name = name;
	}
	virtual void set_message(Span<const char>&& message)
	{
		this->message = message;
	}
	virtual void add_attribute(const Span<const char>& key, const AttributeValue& attr)
	{
		this->m_attrs.emplace(key, attr);
	}

	uint64_t    ts;
	Severity    severity;
	Span<const char> name;
	Span<const char> message;

	typedef Span<const char> key_type;

	std::unordered_map<key_type, AttributeValue,
		std::hash<key_type>, std::equal_to<key_type>,
		Alloc<std::pair<const key_type, AttributeValue>>> m_attrs;
};

class RingBuffer
{
public:
	RingBuffer(size_t len)
	{
		m_seq_r = 0;
		m_seq_w = 0;
		m_buffer = (char*)malloc(len);
		m_read  = m_buffer;
		m_write = m_buffer;
		m_len   = len;
	}

	~RingBuffer()
	{
		free(m_buffer);
	}

	// Get buffer of size n3
	char* reserve(size_t size)
	{
		// round to next 32bit
		size_t padding = (HDR_SIZE-(size&3))&3;
		size_t n = size + padding;  // actaul amount that is needed

		// TODO: LOCK
		lock();


		// while there are items to drop
		while (m_seq_r < m_seq_w)
		{
			// how much space is available?
			size_t avail = available();
			if (avail >= (n + HDR_SIZE*2)) break;
			// need more space..
			drop();
		}

		// not enough space at end?
		if ((m_write + n) > (m_buffer + m_len))
		{
			// wrap around
			m_write = m_buffer;
		}

		char* result = m_write + HDR_SIZE;

		// place TRUE LENGTH and BUSY flag into header to reserve space
		*(int*)m_write = (size + HDR_SIZE) | BUSY_FLAG;		
		// mark next as loop
		*(int*)&m_write[n + HDR_SIZE] = 0;

		m_write += n + HDR_SIZE;

		unlock();

		// return pointer AFTER header
		return result;
	}

	// mark buffer as complete and update counters
	void commit(char* buff)
	{
		int* ptr = (int*)buff;
		// unset busy flag
		ptr[-1] &= ~BUSY_FLAG;

		m_seq_w++; // finally update sequence to indicate new data
	}

	// @brief    Read from buffer.  if DST is too small, data will be truncated and lost
	//           This is meant to be called from 1 thread at a time
	// 
	// @param dst    where to copy data
	// @param avail  Size of dst
	// @returns Number of bytes copied
	size_t consume(char* dst, size_t avail)
	{
		size_t result = 0;

		for (int i = 0; i < 2; i++) // 2 loops, incase first header is 'return to start' len pointer
		{
			if (m_seq_r == m_seq_w) return 0;

			lock();
			uint32_t len = *(int*)m_read;

			if (len & BUSY_FLAG)
			{
				// Shouldn't ever get here.. reading faster than writing?
				unlock();
				return 0;
			}
			if (len == 0) // loop flag?
			{
				m_read = m_buffer;
				unlock();
				continue; // try again
			}

			result = len-HDR_SIZE;
			if (result > avail) result = avail;

			memcpy(dst, m_read+HDR_SIZE, result);
			// round up
			len = ((len + (HDR_SIZE-1)) / HDR_SIZE) * HDR_SIZE;

			m_read += len;
			unlock();

			// loop
			if (m_read >= m_buffer + m_len) m_read = m_buffer;

			m_seq_r++;
			break;
		}

		return result;
	}

private:

	const uint32_t BUSY_FLAG = 0x80000000;
	const size_t HDR_SIZE    = 4;

	std::mutex  m_mutex;
	void lock()
	{
		m_mutex.lock();
		// TODO: platform specific lock
	}

	void unlock()
	{
		// TODO: platform specific lock
		m_mutex.unlock();
	}

	// drop entry.  This should be called
	// with lock held
	void drop()
	{
		if (m_seq_r < m_seq_w)
		{
			uint32_t len = *(int*)m_read;

			// entry is reserved..
			if (len & BUSY_FLAG)
			{
			    //	return;
				// buffer filled up and we need one that is
				// in use... which means another thread is
				// writing to it possibly
				len &= ~BUSY_FLAG;
			}

			if (len == 0) m_read = m_buffer; // loop?
			else
			{
				// round len 
				size_t padding = (HDR_SIZE - (len & 3)) & 3;
				m_read += len + padding;
				// loop
				if (m_read >= m_buffer + m_len) m_read = m_buffer;
				m_seq_r++;
			}
		}
	}
	// calculate available space in a continuous block after m_Write
	// should be called with lock held
	size_t available()
	{
		if (m_write < m_read)
		{
			/// ######------######
			//  write^      ^read

			// available space is difference of pointers
			return m_read - m_write;
		}
		else
		{
			//  -----#############-----
			//  read^       write^
			
			// need space to END of buffer
			size_t a =  m_len - (m_write - m_buffer);
			// or from start
			size_t b = m_read - m_buffer;

			// return larger of 2
			if (b > a) return b;
			return a;
		}
	}

	std::atomic<uint64_t> m_seq_r;
	std::atomic<uint64_t> m_seq_w;
	char*   m_read;
	char*   m_write;
	char*   m_buffer;
	size_t  m_len;
};


class SimpleLogExporter : public LogExporter 
{

	bool full_string = true;

	AllocBackEnd& m_be;
public:

	SimpleLogExporter(AllocBackEnd& be, RingBuffer& buffer)
    : m_be(be)
	, m_record(be)
	, m_buffer(buffer)
	{
	}
	//! Creates a new LogRecord that can be exported
	virtual LogRecord* create_record()
	{
		// only one record at a time is suppported
		return &m_record;
	}

	virtual AttributeList* create_attribute_list()
	{
		return new NoAllocAttributeList(m_be);
	}

	//! Export a record and release resources 
	virtual void export_record(LogRecord* rec)
	{
		Alloc<char> al(m_be);
		std::basic_string<char, std::char_traits<char>, Alloc<char>> foo("abcd", al);

		NoMallocLogRecord* s = (NoMallocLogRecord*)rec;

		// see how much space we need
		size_t req_bytes = measure(*s);

		// get a buffer big enough
		char* buff = m_buffer.reserve(req_bytes);
		char* begin = buff;

		buff = put(buff, s->ts);
		buff = put(buff, s->severity);
		buff = put(buff, s->name);
		buff = put(buff, s->message);

		for (auto it = s->m_attrs.begin();
			it != s->m_attrs.end();
			it++)
		{
			const Span<const char> key = it->first;
			AttributeValue& val = it->second;

			if (!full_string)
			{
				// store as pointer to const string.  This should be a pointer into
				// the string table so no need to copy the string.
				buff = put(buff, key.data);
			}
			else
			{
				buff = put(buff, key);
			}
			buff = put(buff, val);
		}

		m_buffer.commit(begin);

		s->set_message("");
		s->set_name("");
		s->set_timestamp(0);
		// release resources
		s->m_attrs.clear();
	}

	bool read_record(LogRecord& rec, char* buff, size_t len)
	{
		size_t bytes = m_buffer.consume(buff, len);
		if (bytes == 0) return false;

		char* it = buff;
		char* end = it + bytes;

		uint64_t ts;
		Severity sev;
		Span<const char> name;
		Span<const char> message;
		it = get(it, ts);
		it = get(it, sev);
		it = get(it, name);
		it = get(it, message);

		rec.set_timestamp(ts);
		rec.set_severity(sev);
		rec.set_name( std::move(name) );
		rec.set_message( std::move(message) );

		// loop through attributes
		while (it < end)
		{
			Span<const char> attr_name;
			AttributeValue attr(0);

			if (!full_string)
			{
				// Name is stored as pointer to the const string
				const char* name_ptr = 0;
				it = get(it, name_ptr);
				attr_name = name_ptr; // convert to Span
			}
			else
			{				
				it = get(it, attr_name);
			}
			char* st = it;
			it = get(it, attr);

			rec.add_attribute(attr_name, attr);
		}

		return true;
	}

private:

	char* put(char* buff, AttributeValue& val)
	{
		buff = put(buff, val.data_type);
		switch (val.data_type)
		{
		case AttributeType::type_bool: buff = put(buff, val.data.b); break;
		case AttributeType::type_u8:  buff = put(buff, val.data.u8); break;
		case AttributeType::type_s8:  buff = put(buff, val.data.s8); break;
		case AttributeType::type_u16: buff = put(buff, val.data.u16); break;
		case AttributeType::type_s16: buff = put(buff, val.data.s16); break;
		case AttributeType::type_f32: buff = put(buff, val.data.f32); break;
		case AttributeType::type_u32: buff = put(buff, val.data.u32); break;
		case AttributeType::type_s32: buff = put(buff, val.data.s32); break;
		case AttributeType::type_f64: buff = put(buff, val.data.f64); break;
		case AttributeType::type_u64: buff = put(buff, val.data.u64); break;
		case AttributeType::type_s64: buff = put(buff, val.data.s64); break;
		case AttributeType::type_string_view: 
			buff = put(buff, val.data.s ); 
			break;
		}

		return buff;
	}

	// put string pointer
	char* put(char* buff, const char* st)
	{
		memcpy(buff, &st, sizeof(st));
		return buff + sizeof(st);
	}

	char* put(char* buff, Span<char const>&& val)
	{
		return put(buff, val);
	}
	char* put(char* buff, const Span<char const>& val)
	{
		size_t l = val.len;
		if (l > 255) l = 255;

		// 8bit len
		buff = put(buff, (uint8_t)l);

		// text
		memcpy(buff, val.data, l);
		buff += l;
		return buff;
	}

	template<typename T,
		typename std::enable_if<
		   std::is_fundamental< typename std::decay<T>::type >::value ||
		   std::is_enum< typename std::decay<T>::type >::value
		, int>::type = 0
	>
	char* put(char* buff, T&& val)
	{
		memcpy(buff, &val, sizeof(val));
		buff += sizeof(val);
		return buff;
	}

	char* get(char* buff, const char*& val)
	{
		memcpy(&val, buff, sizeof(val));
		return buff + sizeof(val);
	}

	char* get(char* buff, Span<const char>& val)
	{
		val.len = *(uint8_t*)buff;
		val.data = &buff[1];
		buff += val.len + 1;
		return buff;
	}

	char* get(char* buff, AttributeValue& val)
	{
		buff = get(buff, val.data_type);
		switch (val.data_type)
		{
		case AttributeType::type_bool: buff = get(buff, val.data.b); break;
		case AttributeType::type_u8:  buff = get(buff, val.data.u8); break;
		case AttributeType::type_s8:  buff = get(buff, val.data.s8); break;
		case AttributeType::type_u16: buff = get(buff, val.data.u16); break;
		case AttributeType::type_s16: buff = get(buff, val.data.s16); break;
		case AttributeType::type_f32: buff = get(buff, val.data.f32); break;
		case AttributeType::type_u32: buff = get(buff, val.data.u32); break;
		case AttributeType::type_s32: buff = get(buff, val.data.s32); break;
		case AttributeType::type_f64: buff = get(buff, val.data.f64); break;
		case AttributeType::type_u64: buff = get(buff, val.data.u64); break;
		case AttributeType::type_s64: buff = get(buff, val.data.s64); break;
		case AttributeType::type_string_view:
			buff = get(buff, val.data.s);
			break;
		}

		return buff;
	}

	
	template<typename T>
	typename std::enable_if<
		std::is_fundamental< typename std::decay<T>::type >::value ||
		std::is_enum< typename std::decay<T>::type >::value
		,  
		char*>::type get(char* buff, T&& val)
	{
		memcpy(&val, buff, sizeof(val));
		buff += sizeof(val);
		return buff;
	}

	size_t measure(NoMallocLogRecord& rec)
	{
		size_t result = 0;

		result += sizeof(rec.ts);
		result += sizeof(rec.severity);
		result += rec.name.len + 1;   // short string
		result += rec.message.len + 1;  // long string

		for (auto it = rec.m_attrs.begin();
			      it != rec.m_attrs.end();
			      it++)
		{
			Span<const char> key = it->first;
			AttributeValue& val  = it->second;

			if (full_string)
			{
				// 255 length limit
				result += key.len + 1;
			}
			else
			{
				result += sizeof(char*);
			}


			// each type has a 1 byte prefix followed by encoded value..
			// strings also have a length prefix
			result += 1;
			switch (val.data_type)
			{
			case AttributeType::type_bool:
			case AttributeType::type_u8:
			case AttributeType::type_s8:
			   result += 1;
			   break;
			case AttributeType::type_u16:
			case AttributeType::type_s16:
				result += 2;
				break;
			case AttributeType::type_f32:
			case AttributeType::type_u32:
			case AttributeType::type_s32:
				result += 4;
				break;
			case AttributeType::type_f64:
			case AttributeType::type_u64:
			case AttributeType::type_s64:
				result += 8;
				break;
			case AttributeType::type_string_view:
				result += val.data.s.len + 1;
			}
		}

		return result;
	}

	NoMallocLogRecord  m_record;
	RingBuffer&      m_buffer;
};
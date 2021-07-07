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
	// hash for string Spans
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
	virtual ~LogRecord() {}

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

	virtual void set_name(const Span<const char>& name) = 0;
	virtual void set_message(const Span<const char>& message) = 0;
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

	/**
	 Used for scopes.  Creates storage for attributes in
	 a scope
	 */
	virtual AttributeList* create_attribute_list() = 0;
	virtual void release_attribute_list(AttributeList* attrs) = 0;

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
	//! Should create a new record each time, or NULL if no null ones
	//! can be created
	virtual LogRecord* create_record() = 0;

	//! Export a record and release resources 
	virtual void export_record(LogRecord* rec) = 0;

	//! Export range of records and releases resources
	virtual void export_record(LogRecord** begin, size_t len) = 0;
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
	LogScope(LogScope&& other) noexcept;
	~LogScope();
	// explicit end of scope
	void end();
};

class LogProvider;

class Logger
{
public:
	Logger(const char* name, LogProvider& pro)
    : m_provider(pro)
	{
		m_name = name;
	}

	/**
	 Begin a Scope block to attach extra attributes
	 */
	LogScope begin_scope(std::initializer_list<NamedAttribute> at);

	void log(Severity sev, const char* msg, std::initializer_list<NamedAttribute> attrs);

private:
	friend LogScope;

	LogProvider&  m_provider;
	const char*   m_name;

	static thread_local LogScope* m_scope;
};

class LogProvider
{
public:
	Logger& get(const char* name)
	{
		// holds lock while in the function
		std::lock_guard<std::mutex> guard(m_lock);

		auto it = m_loggers.find(name);

		// was in the map
		if (it != m_loggers.end()) return *it->second;

		Logger* logger = new Logger(name, *this);
		m_loggers[name] = logger;

		return *logger;
	}

	LogProcessor* get_processor() { return m_proc; }
	LogExporter* get_exporter() { return m_exp;  }

	LogProvider& with_processor(LogProcessor* proc)
	{
		m_proc = proc;

		return *this;
	}
	LogProvider& with_exporter(LogExporter* exp)
	{
		m_exp = exp;
		return *this;
	}
protected:

	LogProcessor* m_proc;
	LogExporter* m_exp;

	std::unordered_map<const char*, Logger*> m_loggers;
	std::mutex  m_lock;  //!< protects m_loggers
};

struct AllocBackEnd
{
	static const int COUNT = 2048;
	static const int BUFFERS = 4;
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

		bool operator==(const SeqCounter& rhs) const
		{
			return (index == rhs.index)
				&& (seq == rhs.seq)
				;
		}
	};
	std::atomic<SeqCounter> free_list[4];
	size_t size[4];

	int MAX[4];

	AllocBackEnd()
	{
		memset(MAX, 0, sizeof(MAX));

		size[0] = 16;
		size[1] = 64;
		size[2] = 128;
		size[3] = 256;


		for (int i = 0; i < BUFFERS; i++)
		{
			free_list[i] = { 0,0 };
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
			p[(COUNT - 1) * size[k] / sizeof(int)] = -1;
		}
	}

	void* alloc(size_t n)
	{
		void* rst = 0;

		// find the smallest size that meets the requested size
		for (int i = 0; i < BUFFERS; i++)
		{
			if (n > size[i]) continue;

			uint32_t* next = 0;

			SeqCounter nxt;
			do
			{
				rst = 0;  // result pointer

				// latest free list pointer
				SeqCounter curr = free_list[i];

				if (curr.index > COUNT)
					break;  // empty free list

				next = (uint32_t*)&(buff[i][size[i] * (curr.index)]);

				assert((char*)next >= buff[i]);
				int last = *next;
				if ((*next >= COUNT) && (free_list[i].load() ==  curr))
				{
					while (true);
					break;  // empty free list
				}

				// another thread pulls item out of free list
				// and inserts it again here..  *next would
				// now be wrong, as the buffer has changed

				nxt.index = *next;
				nxt.seq = curr.seq + 1;

				if (nxt.index >= COUNT)
				{
					while (free_list[i].load().index == curr.index);
				}

				// if free_list changed, loop again
				if (!free_list[i].compare_exchange_weak(curr, nxt)) continue;

				if (nxt.index >= COUNT)
					while (true);

				if (MAX[i] < nxt.index)
				{
					MAX[i] = nxt.index;
					if (i == 1)
						printf("MX %d\n", MAX[i]);
				}

				break;
			} while (true);

			rst = next;
			break;
		}

		assert(rst != 0);
		return rst;
	}

	void dealloc(char* ptr)
	{
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

					if (next.index >= COUNT)
						break;  // empty free list

					idx.seq = next.seq + 1;
				} while (!free_list[i].compare_exchange_weak(next, idx));

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
  Does not allocate memory.  Uses AllocBackEnd to get
  memory it needs.   Allows for quick building of record and in places
  that may not allow memory to be allocated
 */
class NoMallocLogRecord : public LogRecord
{
	AllocBackEnd& m_be;
public:
	NoMallocLogRecord(AllocBackEnd& be)
	: m_be(be)
	, m_attrs(Alloc< decltype(m_attrs)::value_type>(be))
	{
		m_attrs.reserve(8);
		ts = 0;
		severity = Severity::Trace;
		name.len = 0;
		message.len = 0;
	}

	~NoMallocLogRecord()
	{
		// free all the strings
		for (auto it = m_attrs.begin();
			it != m_attrs.end();
			it++)
		{
			if (it->second.data_type == AttributeType::type_string_view)
			{
				m_be.dealloc((char*)it->second.data.s.data);
			}
		}
	}
	virtual void set_timestamp(uint64_t nsec)
	{
		ts = nsec;
	}
	virtual void set_severity(Severity severity)
	{
		this->severity = severity;
	}

	// bring in the overloaded names that take const char*
	using LogRecord::set_message;
	using LogRecord::set_name;

	virtual void set_name(const Span<const char>& name)
	{
		this->name = name;
	}
	virtual void set_message(const Span<const char>& message)
	{
		this->message = message;
	}
	virtual void set_name(Span<const char>&& name)
	{
		this->name = std::move(name);
	}

	virtual void set_message(Span<const char>&& message)
	{
		this->message = std::move(message);
	}

	virtual void add_attribute(const Span<const char>& key, const AttributeValue& attr)
	{
		const auto& pair = this->m_attrs.emplace(key.data, attr);

		// duplicate string, so it can go out of scope
		if (attr.data_type == AttributeType::type_string_view)
		{
			AttributeValue& back = pair.first->second;
			back.data.s.data = (char*)m_be.alloc(attr.data.s.len);
			memcpy((char*)back.data.s.data, attr.data.s.data, attr.data.s.len);
		}
	}

	uint64_t    ts;
	Severity    severity;
	Span<const char> name;
	Span<const char> message;

	typedef const char* key_type;

	std::unordered_map<key_type, AttributeValue,
		std::hash<key_type>, std::equal_to<key_type>,
		Alloc<std::pair<const key_type, AttributeValue>>> m_attrs;
};

class NoAllocAttributeList : public AttributeList
{
public:
	std::vector<NamedAttribute, Alloc<NamedAttribute>> m_attrs;

	NoAllocAttributeList(AllocBackEnd& be)
		: m_attrs(Alloc<NamedAttribute>(be))
	{
		m_attrs.reserve(4);
	}

	~NoAllocAttributeList() {}

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

template<typename T>
class CirculeBuffer
{
public:
	CirculeBuffer(size_t len)
	{
		m_capacity = len;
		m_data = new std::atomic<T*>[len];
		m_read = 0;
		m_write = 0;
	}

	void add(T* ptr)
	{
		while (true)
		{
			// grab local copy
			uint64_t w = m_write;
			if (size() >= m_capacity-1)
			{
				continue;
				//while (true); // out of space?!
			}

			uint64_t index = w % m_capacity;

			// expect to be empty, if not, another thread
			// got here first, so loop again
			T* exp = 0;
			if (!m_data[index].compare_exchange_weak(exp, ptr)) continue;

		    // update write pointer
			// The exchange could fail, if a new item was added and 
			// read between the first lines of this loop and here.
			// The prior compare/exchange would pass becuase the read
			// leaves m_data null,  this m_write however would reflect
			// the added item(s).  So loop on fail
			if (!m_write.compare_exchange_weak(w, w + 1))
			{
				// change it back to null since it should be empty..
				// and loop to try again
				m_data[index] = nullptr;
				continue;
			}

			// the index did not change so everything was successful
			break;
		}
	}

	T* consume()
	{
		if (m_write == m_read)
		{
			return nullptr;
		}

		uint64_t index = m_read % m_capacity;
		T* exp = m_data[index];
		while (!m_data[index].compare_exchange_weak(exp, nullptr) )
		{
			index = m_read % m_capacity;
			exp = m_data[index];
		}

		// increment
		m_read++;

		return exp;
	}

	size_t size()
	{
		return m_write - m_read;
	}
private:
	std::atomic<uint64_t> m_read;
	std::atomic<uint64_t> m_write;
	size_t m_capacity;
	std::atomic<T*>* m_data;
};

class SimpleLogProcessor : public LogProcessor
{
	typedef NoMallocLogRecord RecordType;
	typedef Alloc<RecordType>::rebind<NoAllocAttributeList>::other AllocAttrList;
public:
	SimpleLogProcessor(LogProvider& p, AllocBackEnd& be)
	: m_be(be)
	, m_alloc(be)
    , m_buffer(128)
    , m_work_thread(&SimpleLogProcessor::DoWork, this)
	, m_provider(p)
	{

	}

	void DoWork()
	{
		while (true)
		{
			size_t count = m_buffer.size();

			// TODO: limit count to a max value?

			std::vector<LogRecord*> vec;
			vec.reserve(count); // resever the number of entries needed

			// get reference here to ensure it doesn't change during loop
			LogExporter* exp = m_provider.get_exporter();
			for (size_t i = 0; i<count; i++)
			{
				// translate from the RecordType type
				// to whatever the exporter uses
				RecordType* rec = m_buffer.consume();
				LogRecord* r = exp->create_record();
				if (r == nullptr) break;  // exporter is out of records?

				r->set_message(rec->message.data);
				r->set_name(rec->name.data);
				r->set_severity(rec->severity);
				r->set_timestamp(rec->ts);

				for (auto it = rec->m_attrs.begin();
					it != rec->m_attrs.end();
					it++)
				{
					r->add_attribute(it->first, it->second);
				}

				rec->~RecordType();
				m_alloc.deallocate((RecordType*)rec, 1);

				vec.push_back(r);
			}

			// no buffering, pass directly on
			exp->export_record(vec.data(), vec.size());
		}
	}

	virtual AttributeList* create_attribute_list()
	{

		AllocAttrList a = m_alloc;
		void* p = a.allocate(1);
		return new (p) NoAllocAttributeList(m_be);
	}

	virtual void release_attribute_list(AttributeList* attrs)
	{
		AllocAttrList a = m_alloc;
		attrs->~AttributeList();
		a.deallocate((NoAllocAttributeList*)attrs, 1);
	}

	//! Creates new LogRecord to be filled by Logger
	virtual LogRecord* create_record()
	{
		void* ptr = m_alloc.allocate(1);
		RecordType* slr = new (ptr) RecordType(m_be);
		return slr;
	}

	/**
	Adds a record to be processed.  Processor will
	release resources for the record
	*/
	virtual void add_record(LogRecord* rec)
	{
		// add to circular buffer
		m_buffer.add((RecordType*)rec);
	}

private:

	AllocBackEnd&     m_be;
	Alloc<RecordType> m_alloc;
	CirculeBuffer<RecordType> m_buffer;
	std::thread       m_work_thread;
	LogProvider&      m_provider;

};

/**
 Implements a LogRecord using an unordered_map and duplicates
 strings.  Allowing the source strings to be released. Recommend
 to use this on the exporter backend only
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

	~SimpleLogRecord()
	{
		reset();
	}

	void reset()
	{
		reset(name);
		reset(message);
		for (auto it = m_attrs.begin();
			it != m_attrs.end();
			it++)
		{
			if (it->second.data_type == AttributeType::type_string_view)
			{
				reset(it->second.data.s);
			}
		}
		m_attrs.clear();
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

	virtual void set_name(const Span<const char>& name)
	{
		clone(this->name, name);
	}
	virtual void set_message(const Span<const char>& message)
	{
		clone(this->message, message);
	}

	virtual void set_name(Span<const char>&& name)
	{
		clone(this->name, name);
	}
	virtual void set_message(Span<const char>&& message)
	{
		clone(this->message, message);
	}

	virtual void add_attribute(const Span<const char>& key, const AttributeValue& attr)
	{
		const auto& back = this->m_attrs.emplace(key, attr);
		if (attr.data_type == AttributeType::type_string_view)
		{
			auto& av = back.first->second;
			av.data.s.len = 0;
			av.data.s.data = 0;
			clone(av.data.s, attr.data.s);		
		}
	}

	uint64_t         ts;
	Severity         severity;
	Span<const char> name;
	Span<const char> message;

	typedef std::string key_type;

	std::unordered_map<key_type, AttributeValue> m_attrs;
protected:

	void reset(Span<const char>& src)
	{
		free((char*)src.data);
		src.data = 0;
		src.len = 0;
	}

	void clone(Span<const char>& dst, const Span<const char>& other)
	{
		if (dst.len != other.len)
		{
			if (dst.data) free((char*)dst.data);
			dst.data = (char*)malloc(other.len * sizeof(char));
			dst.len = other.len;
		}
		assert(dst.data != 0);
		memcpy((char*)dst.data, other.data, other.len);
	}
};

/**
 A Ring buffer supporting entries of arbatrary size.  Each is prefixed
 with a 32bit length.  If no space available at the end, older entries are
 dropped until there is enough space available
 */
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

/**
 print attribute value using printf
 */
struct AttrPrinter
{
	void operator()(bool b)
	{
		printf("%d", b);
	}

	void operator()(float b)
	{
		printf("%f", b);
	}
	void operator()(double b)
	{
		printf("%lf", b);
	}
	void operator()(int8_t b)
	{
		printf("%d", b);
	}
	void operator()(int16_t b)
	{
		printf("%d", b);
	}
	void operator()(int32_t b)
	{
		printf("%d", b);
	}
	void operator()(int64_t b)
	{
		printf("%lld", b);
	}

	void operator()(uint8_t b)
	{
		printf("%u", b);
	}
	void operator()(uint16_t b)
	{
		printf("%u", b);
	}
	void operator()(uint32_t b)
	{
		printf("%u", b);
	}
	void operator()(uint64_t b)
	{
		printf("%llu", b);
	}

	void operator()(const Span<const char>& b)
	{
		printf("%.*s", (int)b.len, b.data);
	}
};

/**
 'Export' LogRecord using printf
 */
class PrintfLogExporter : public LogExporter 
{
public:
	PrintfLogExporter(AllocBackEnd& be)
	{
	}
	//! Creates a new LogRecord that can be exported
	virtual LogRecord* create_record()
	{
		// TODO: use a pool of records..
		return new SimpleLogRecord();
	}

	virtual void export_record(LogRecord** begin, size_t len)
	{
		for (size_t i = 0; i < len; i++)
		{
			export_record(begin[i]);
		}
	}

	//! Export a record and release resources 
	virtual void export_record(LogRecord* rec)
	{
		SimpleLogRecord* s = (SimpleLogRecord*)rec;

		printf("name: %.*s\n", (int)s->name.len, s->name.data);
		printf(" msg: %.*s\n", (int)s->message.len, s->message.data);
		printf("  ts: %llu\n", s->ts);
		printf(" sev: %d\n", s->severity);
		for (auto it = s->m_attrs.begin();
			      it != s->m_attrs.end();
			      it++)
		{
			printf("\t%s : ", it->first.c_str());
			AttrPrinter ptr;
			visit(ptr, it->second);
			printf("\n");
		}
		
		// release
		delete rec;
	}

private:

	SimpleLogRecord  m_record;
};
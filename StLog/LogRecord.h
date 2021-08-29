#pragma once
#include <chrono>
#include <unordered_map>
#include <atomic>
#include <assert.h>
#include <mutex>
#include <string.h>
#include <vector>

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

struct TimeHelper
{
    // Converts from nanosec offset from 1970 epoc to
    // iso8601 string with timezone
    // @param ts_buff should be at least 35 bytes
    static void ToIso8601(uint64_t nsec, char* ts_buff, size_t buff_len)
    {
        std::chrono::nanoseconds ts(nsec);
        std::chrono::system_clock::duration d = std::chrono::duration_cast<std::chrono::system_clock::duration>(ts);
        std::chrono::system_clock::time_point tp(d);

        // nsec part
        uint64_t fractional = nsec % 1000000000;

        // convert from systemtime into ISO8601 format
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
        std::tm* tm = localtime(&tt);
        size_t len = strftime(ts_buff, buff_len, "%FT%T", tm);

        // compute the time in gmt and local, and get the difference
        std::tm gm_ = *gmtime(&tt);
        std::tm loc_ = *localtime(&tt);
        // correctly compute DST..  the tm values
        // are incorrect for somereason
        gm_.tm_isdst = -1;
        loc_.tm_isdst = -1;
        // convert back to seconds.  tm_isdst is set to -1 to force mktime
        // to calculate the correct DST value.
        time_t gm = mktime(&gm_);
        time_t loc = mktime(&loc_);
        time_t diff = loc - gm;  // difference in seconds

        // compute tz offsets

        // timezone offset
        int8_t tz_hour = diff / 3600;
        int8_t tz_min = (diff % 3600) / 60;

        // append nsec and TZ offset into ISO8601 string
        sprintf(ts_buff + len, ".%09llu%+03d:%02d", fractional, tz_hour, tz_min);
    }
};

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


// helper to tell the difference between long and long long types
// on various platforms
template<size_t N, bool SIGNED = false>
AttributeType get_attr_type()
{
    static_assert((N != 4) && (N != 8), "unsupported type");
}
template<> AttributeType get_attr_type<8, false>() { return AttributeType::type_u64; }
template<> AttributeType get_attr_type<8, true>() { return AttributeType::type_s64; }
template<> AttributeType get_attr_type<4, false>() { return AttributeType::type_u32; }
template<> AttributeType get_attr_type<4, true>() { return AttributeType::type_s32; }


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

    AttributeValue(unsigned long val)
    {
        data.u64 = val;
        data_type = get_attr_type<sizeof(val), false>();
    }

    AttributeValue(unsigned long long val)
    {
        data.u64 = val;
        data_type = get_attr_type<sizeof(val), false>();
    }

    AttributeValue(signed long val)
    {
        data.u64 = val;
        data_type = get_attr_type<sizeof(val), true>();
    }

    AttributeValue(signed long long val)
    {
        data.u64 = val;
        data_type = get_attr_type<sizeof(val), true>();
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

    virtual void set_timestamp(int64_t nsec) = 0;
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

    virtual void add_attribute(const char* key, const AttributeValue& attr) = 0;
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

class NopLogProcess : public LogProcessor
{
public:
    //! Creates new LogRecord to be filled by Logger
    virtual LogRecord* create_record() { return nullptr;  }

    /**
     Used for scopes.  Creates storage for attributes in
     a scope
     */
    virtual AttributeList* create_attribute_list() { return nullptr; }
    virtual void release_attribute_list(AttributeList* attrs)
    {
        // do nothing..?
    }

    /**
    Adds a record to be processed.  Processor will
    release resources for the record
    */
    virtual void add_record(LogRecord* rec)
    {
        // do nothing
    }
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

class NopLogExporter : public LogExporter
{
public:
    virtual LogRecord* create_record() { return nullptr; }

    //! Export a record and release resources 
    virtual void export_record(LogRecord* rec)
    {
    }

    //! Export range of records and releases resources
    virtual void export_record(LogRecord** begin, size_t len)
    {
    }
};

class Logger;

class LogScope
{
    friend Logger;
    AttributeList* attrs;
    LogScope* next;
    Logger* logger;
    LogProcessor* pro;

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
    LogProvider()
    {
        m_proc = &m_nop_proc;
        m_exp = &m_nop_exp;
    }

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

    LogProcessor* get_processor() 
    { 
        return m_proc;
    }
    LogExporter* get_exporter() { return m_exp;  }

    LogProvider& with_processor(LogProcessor* proc)
    {
        if (proc == nullptr) proc = &m_nop_proc;
        m_proc = proc;

        return *this;
    }
    LogProvider& with_exporter(LogExporter* exp)
    {
        m_exp = exp;
        return *this;
    }
protected:

    NopLogProcess  m_nop_proc;
    NopLogExporter m_nop_exp;
    LogProcessor* m_proc;
    LogExporter* m_exp;

    std::unordered_map<const char*, Logger*> m_loggers;
    std::mutex  m_lock;  //!< protects m_loggers
};

struct AllocBackEnd
{
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

    struct Buffer
    {
        size_t size;    //!< bytes per entry
        size_t count;	//!< total number of entries
        char*  buffer;  //!< start of buffer.  end is buffer + size*count
        char*  end;
        int    MAX;
    };
    
    std::atomic<SeqCounter>* free_list;
    Buffer*                  m_buffer;
    size_t                   m_count;

    AllocBackEnd( std::initializer_list<Buffer> buffers)
    {
        m_count   = buffers.end() - buffers.begin();

        m_buffer = new Buffer[m_count];		
        free_list = new std::atomic<SeqCounter>[m_count];

        for (int i = 0; i < m_count; i++)
        {
            free_list[i] = { 0,0 };
        }

        // build free list
        for (size_t k = 0; k < m_count; k++)  // each size
        {
            Buffer& b = m_buffer[k];
            b = buffers.begin()[k];
            b.end = b.buffer + b.size * b.count;
            b.MAX = 0;

            int* p = (int*)b.buffer;
            assert(p != 0);

            // each block in buffer, point to next
            for (size_t i = 0; i < b.count; i++)
            {
                p[i * b.size / sizeof(int)] = i + 1;
            }

            // last pointer is invalid
            p[(b.count - 1) * b.size / sizeof(int)] = -1;
        }
    }

    void* alloc(size_t n)
    {
        void* rst = 0;

        // find the smallest size that meets the requested size
        for (int i = 0; i < m_count; i++)
        {
            Buffer& b = m_buffer[i];

            if (n > b.size) continue;

            uint32_t* next = 0;

            SeqCounter nxt;
            do
            {
                rst = 0;  // result pointer

                // latest free list pointer
                SeqCounter curr = free_list[i];

                if (curr.index > b.count)
                    break;  // empty free list

                next = (uint32_t*)&(b.buffer[b.size * (curr.index)]);

                // not before start of buffer
                assert((char*)next >= b.buffer);

                int last = *next;
#ifdef DEBUG_ALLOC
                if ((*next >= b.count) && (free_list[i].load() ==  curr))
                {
                    while (true);
                    break;  // empty free list
                }
#endif
                // another thread pulls item out of free list
                // and inserts it again here..  *next would
                // now be wrong, as the buffer has changed

                nxt.index = *next;
                nxt.seq = curr.seq + 1;

#ifdef DEBUG_ALLOC
                if (nxt.index >= b.count)
                {
                    while (free_list[i].load().index == curr.index);
                }
#endif
                // if free_list changed, loop again
                if (!free_list[i].compare_exchange_weak(curr, nxt)) continue;

#ifdef DEBUG_ALLOC
                if (nxt.index >= b.count)
                    while (true);
#endif

                if (b.MAX < nxt.index)
                {
                    b.MAX = nxt.index;
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
        for (int i = 0; i < m_count; i++)
        {
            Buffer& b = m_buffer[i];

            char* begin = b.buffer;
            char* end = b.end; // begin + b.count * b.size;  // end of allocated range

            // in allocated memory range?
            if (ptr >= begin && ptr < end)
            {
                SeqCounter idx;
                idx.index = (uint16_t)((ptr - begin) / b.size );  // block number;

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
#if 0
                    if (next.index >= b.count)
                        break;  // empty free list
#endif
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
            if (it->value.data_type == AttributeType::type_string_view)
            {
                m_be.dealloc((char*)it->value.data.s.data);
            }
            /*
            if (it->second.data_type == AttributeType::type_string_view)
            {
                m_be.dealloc((char*)it->second.data.s.data);
            }
            */
        }
    }
    virtual void set_timestamp(int64_t nsec)
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

    virtual void add_attribute(const char* key, const AttributeValue& attr)
    {
        this->m_attrs.emplace_back( NamedAttribute{ key, attr } );
        if (attr.data_type == AttributeType::type_string_view)
        {
            NamedAttribute& back = m_attrs.back();
            back.value.data.s.data = (char*)m_be.alloc(attr.data.s.len);
            memcpy((char*)back.value.data.s.data, attr.data.s.data, attr.data.s.len);
        }

        /*
        const auto& pair = this->m_attrs.emplace(key, attr);

        // duplicate string, so it can go out of scope
        if (attr.data_type == AttributeType::type_string_view)
        {
            AttributeValue& back = pair.first->second;
            back.data.s.data = (char*)m_be.alloc(attr.data.s.len);
            memcpy((char*)back.data.s.data, attr.data.s.data, attr.data.s.len);
        }
        */
    }

    uint64_t    ts;
    Severity    severity;
    Span<const char> name;
    Span<const char> message;

    typedef const char* key_type;

    std::vector<NamedAttribute, Alloc<NamedAttribute> > m_attrs;
    /*
    using a map is MUCH slower than a vector
    std::unordered_map<key_type, AttributeValue,
        std::hash<key_type>, std::equal_to<key_type>,
        Alloc<std::pair<const key_type, AttributeValue>>> m_attrs;
    */
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

    T* add(T* ptr)
    {
        while (true)
        {
            // grab local copy
            uint64_t w = m_write;
            uint64_t r = m_read;
            if ((w-r) >= m_capacity)
            {
                // drop oldest..				
                return consume();
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
        return nullptr;
    }

    T* consume()
    {
        // increment m_read to ensure only this thread
        // will modify this index
        uint64_t r;
        uint64_t w;
        uint64_t index;
        do
        {
            // load read before write to allow write
            // to increment
            r = m_read;
            w = m_write;
            index = r % m_capacity;

            // check for empty before compare and exchange
            if (w == r) return nullptr;
        } while (!m_read.compare_exchange_weak(r, r + 1));

        // at this point this will be the only thread accessing
        // this index, so its safe to write it directly
        T* exp = m_data[index];
        m_data[index] = nullptr;

        /* for testing, can do a compare and 
        while (!m_data[index].compare_exchange_weak(exp, nullptr) )
        {
            // should never get here
            assert(0);
        }
        assert(exp != nullptr);
        */
        return exp;
    }

    size_t size()
    {
        size_t diff;
        uint64_t w;
        do
        {
            w = m_write.load();
            diff = w - m_read;

            // if write changed, loop.  Its possible
            // for write and read to update making
            // m_read > w
        } while (!m_write.compare_exchange_weak(w, w));
        return diff;
    }
    std::atomic<uint64_t> m_read;
    std::atomic<uint64_t> m_write;
private:
    size_t m_capacity;
    std::atomic<T*>* m_data;

    // TODO: make this a spinlock on supported platform
    std::mutex m_read_lock;
};

class NoAllocLogProcessor : public LogProcessor
{
    typedef NoMallocLogRecord RecordType;
    typedef Alloc<RecordType>::rebind<NoAllocAttributeList>::other AllocAttrList;

    size_t THRESHOLD;
    size_t BUFFER_SIZE;
    std::chrono::milliseconds  SLEEP_TIME;
public:

    struct Options
    {
        size_t  buffer_size = 128;		//!< Size of ring buffer in entries
        size_t  threshold   = 64;		//!< how full buffer should be, before forced flush
        //! max time between flushes of buffer
        const std::chrono::milliseconds timeout = std::chrono::milliseconds(100);
    };
    NoAllocLogProcessor(LogProvider& p, AllocBackEnd& be, const Options& opt = {})
    : m_be(be)
    , m_alloc(be)
    , m_buffer(opt.buffer_size)
    , m_work_thread(&NoAllocLogProcessor::DoWork, this)
    , m_provider(p)
    {
        THRESHOLD = opt.threshold;
        BUFFER_SIZE = opt.buffer_size;
        SLEEP_TIME = opt.timeout;

        if (THRESHOLD > BUFFER_SIZE) THRESHOLD = BUFFER_SIZE;
    }

    ~NoAllocLogProcessor()
    {
        shutdown();
        m_work_thread.join();
    }

    /**
     Worker thread that pulls data from buffer and sends to exporter.
     This will pull off as many as it can at once and forward to the
     exporter in batches. 
     */
    void DoWork()
    {
        std::chrono::milliseconds timeout = SLEEP_TIME; // defualt
        while (m_shutdown == false)
        {
            //m_buffer_cv.wait_for(timeout);
            auto t1 = std::chrono::system_clock::now();

            export_records();

            auto t2 = std::chrono::system_clock::now();

            timeout = std::chrono::duration_cast<std::chrono::milliseconds>(SLEEP_TIME - (t2 - t1));
            
            // took longer than the SLEEP_TIME, so don't sleep for next cycle
            if (timeout.count() < 0)
            {
                timeout = std::chrono::milliseconds::zero();
            }
        }

        // drain any remaining
        // need to keep looking while add_record is called, incase
        // one was received while this was draining
        while ( (m_buffer.size() > 0) || m_busy)
        {
            export_records();
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

    int dropped = 0;
    /**
    Adds a record to be processed.  Processor will
    release resources for the record
    */
    virtual void add_record(LogRecord* rec)
    {
        if (m_shutdown)
        {
            // don't push to buffer, but free resources
            rec->~LogRecord();
            m_alloc.deallocate((RecordType*)rec, 1);
            return;
        }

        m_busy = true;

        LogRecord* lost = nullptr;		
        do 
        {
            // add to circular buffer
            lost = m_buffer.add((RecordType*)rec);

            // had to drop one?
            if (lost!=nullptr)
            {
                dropped++;
                lost->~LogRecord();
                m_alloc.deallocate((RecordType*)lost, 1);
            }
        } while (lost!=nullptr);
        // Wake up worker thread if buffer is
        // above theshold full
        if (m_buffer.size() >= THRESHOLD)
            m_buffer_cv.notify_one();

        m_busy = false;
    }

    void shutdown()
    {
        m_shutdown = true;
    }
private:

    void export_records()
    {
        // get reference here to ensure exporter doesn't change during loop
        LogExporter* exp = m_provider.get_exporter();

        std::vector<LogRecord*> vec;
        size_t count = m_buffer.size(); // TODO: limit count to a max value?

        vec.reserve(count); // resever the number of entries needed

        for (size_t i = 0; i < count; i++)
        {
            LogRecord* r = exp->create_record();
            if (r == nullptr) break;  // exporter is out of records?

            // translate from the RecordType type
            // to whatever the exporter uses
            RecordType* rec = m_buffer.consume();

            r->set_message(rec->message.data);
            r->set_name(rec->name.data);
            r->set_severity(rec->severity);
            r->set_timestamp(rec->ts);

            for (auto it = rec->m_attrs.begin();
                it != rec->m_attrs.end();
                it++)
            {
                //r->add_attribute(it->first, it->second);
                r->add_attribute(it->name, it->value);
            }

            rec->~RecordType();
            m_alloc.deallocate((RecordType*)rec, 1);

            vec.push_back(r);
        }

        dropped = 0;

        exp->export_record(vec.data(), vec.size());
    }

    AllocBackEnd&     m_be;
    Alloc<RecordType> m_alloc;
    CirculeBuffer<RecordType> m_buffer;
    std::thread        m_work_thread;
    LogProvider&       m_provider;
    std::atomic<bool>  m_shutdown;
    std::atomic<bool>  m_busy;   //<! while 'add_record'
    /**
     Condition Variable helper to take lock 
     */
    struct ConditionVar
    {
        std::condition_variable cv;
        std::mutex              lock;

        template<class Rep, class Period>
        void wait_for(std::chrono::duration<Rep, Period>& timeout)
        {
            std::unique_lock<std::mutex> l(lock);
            cv.wait_for(l, timeout);
        }

        // wake up other thread
        void notify_one()
        {
            cv.notify_one();
        }
    };

    ConditionVar m_buffer_cv;

};

/**
 Implements a LogRecord using an unordered_map with std::string keys
 and duplicates string attributes.  Allowing the source strings to be released. 
 Recommend to use this on the exporter backend only, as it allocates memory
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

    virtual void set_timestamp(int64_t nsec)
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

    virtual void add_attribute(const char* key, const AttributeValue& attr)
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

    int64_t          ts;
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
 print attribute value using printf
 */
struct AttrPrinter
{
    FILE* dst;
    AttrPrinter(FILE* f)
    {
        dst = f;
    }
    void operator()(bool b)
    {
        fprintf(dst,"%d", b);
    }

    void operator()(float b)
    {
        fprintf(dst, "%f", b);
    }
    void operator()(double b)
    {
        fprintf(dst, "%lf", b);
    }
    void operator()(int8_t b)
    {
        fprintf(dst, "%d", b);
    }
    void operator()(int16_t b)
    {
        fprintf(dst, "%d", b);
    }
    void operator()(int32_t b)
    {
        fprintf(dst, "%d", b);
    }
    void operator()(int64_t b)
    {
        fprintf(dst, "%lld", b);
    }

    void operator()(uint8_t b)
    {
        fprintf(dst, "%u", b);
    }
    void operator()(uint16_t b)
    {
        fprintf(dst, "%u", b);
    }
    void operator()(uint32_t b)
    {
        fprintf(dst, "%u", b);
    }
    void operator()(uint64_t b)
    {
        fprintf(dst, "%llu", b);
    }

    void operator()(const Span<const char>& b)
    {
        fprintf(dst, "%.*s", (int)b.len, b.data);
    }
};

/**
 'Export' LogRecord using printf
 */
class PrintfLogExporter : public LogExporter 
{
    std::vector<SimpleLogRecord*> logPool;
public:
    PrintfLogExporter(FILE* dst)
    {
        m_dst = dst;

        for (int i = 0; i < 128; i++)
            logPool.push_back(new SimpleLogRecord());
    }

    ~PrintfLogExporter()
    {
        for (int i = 0; i < logPool.size(); i++)
        {
            delete logPool[i];
        }
    }
    //! Creates a new LogRecord that can be exported
    virtual LogRecord* create_record()
    {
        if (logPool.size() == 0) return 0;

        LogRecord* front = logPool.back();
        logPool.pop_back();
        return front;
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

        fprintf(m_dst,"name: %.*s\n", (int)s->name.len, s->name.data);
        fprintf(m_dst, " msg: %.*s\n", (int)s->message.len, s->message.data);

        char ts_buff[48];
        TimeHelper::ToIso8601(s->ts, ts_buff, sizeof(ts_buff));

        fprintf(m_dst, "  ts: %s\n", ts_buff);

        fprintf(m_dst, " sev: %d\n", s->severity);
        for (auto it = s->m_attrs.begin();
                  it != s->m_attrs.end();
                  it++)
        {
            fprintf(m_dst, "\t%s : ", it->first.c_str());
            AttrPrinter ptr(m_dst);
            visit(ptr, it->second);
            fprintf(m_dst, "\n");
        }
        
        // release
        s->reset();
        logPool.push_back(s);
    }

private:

    FILE* m_dst;
};
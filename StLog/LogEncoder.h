#pragma once
#include <memory>
#include "ObjectValue.h"
#include <string_view>
#include <assert.h>

struct LoggerEntryHeader
{
	uint16_t level;
	uint64_t ts;
	std::string_view message;
};

class LoggerBuffer;

struct LogDecoder
{
	const char* buffer;
	size_t len;

	LogDecoder(const char* b, size_t l)
	{
		buffer = b;
		len = l;
	}

	void args(int& num)
	{
		num = *buffer++;
	}

	template<typename T, 
		typename = typename std::enable_if<
		std::is_same<Field,T>::value
	>::type >
	Field decode()
	{
		Field f = Field(std::string_view(&buffer[1], buffer[0]), 0);
		buffer += buffer[0] + 1;
		(*this)(f.value);
		return f;
	}

	template<typename T, 
		typename = typename std::enable_if<
		   std::is_integral<T>::value
		|| std::is_floating_point<T>::value
	    >::type >
    T decode()
	{
		T v;
		memcpy(&v, buffer, sizeof(v));
		buffer += sizeof(v);
		return v;
	}

	void operator()(ParamValue& v)
	{
		switch ( (ArgType )*buffer++)
		{
		case ArgType::i8Type:  v = decode<int8_t>(); break;
		case ArgType::u8Type:  v = decode<uint8_t>(); break;
		case ArgType::i16Type: v = decode<int16_t>(); break;
		case ArgType::u16Type: v = decode<uint16_t>(); break;
		case ArgType::i32Type: v = decode<int32_t>(); break;
		case ArgType::u32Type: v = decode<uint32_t>(); break;
		case ArgType::i64Type: v = decode<int64_t>(); break;
		case ArgType::u64Type: v = decode<uint64_t>(); break;
		case ArgType::float32Type: v = decode<float>(); break;
		case ArgType::float64Type: v = decode<double>(); break;
		case ArgType::stringType: 
			v = std::string_view(&buffer[1], buffer[0]);
			buffer += buffer[0] + 1;
			break;
		}
	}
	void operator()(LoggerEntryHeader& v)
	{
		memcpy(&v.level, buffer, sizeof(v.level)); buffer += sizeof(v.level);
		memcpy(&v.ts, buffer, sizeof(v.ts));       buffer += sizeof(v.ts);
		
		v.message = std::string_view(&buffer[1], buffer[0]);
		buffer += 1 + buffer[0];
	}
};

struct LogEncoder
{
	char* buffer;
	char* end;
	bool fail;

	LoggerBuffer* lb;
	LogEncoder(LoggerBuffer* lb, char* b, char* e)
	{
		this->lb = lb;
		fail = false;
		buffer = b;
		end = e;
	}

	void commit();


	void args(size_t n)
	{
		*buffer++ = n;
	}

	/*
	std::string_view,
		std::shared_ptr<ObjectValue>, // map of values
		std::vector<ParamValue>
	*/
	void operator()(const std::shared_ptr<ObjectValue>& v)
	{
		buffer[0] = (uint8_t)ArgType::objectType;
		for (int i = 0; i < v->data.size(); i++)
		{
			(*this)(v->data[i]);
		}
	}
	void operator()(const std::vector<ParamValue>& v)
	{
	}

	void operator()(bool v)
	{
	}

	void operator()(LoggerEntryHeader& v)
	{
		const size_t NEEDED = 0
			+ sizeof(v.level)
			+ sizeof(v.ts)
			+ v.message.size()+1
			;

		if (fail || buffer + NEEDED >= end)
		{
			fail = true;
			return;
		}

		memcpy(buffer, &v.level, sizeof(v.level));  buffer += sizeof(v.level);
		memcpy(buffer, &v.ts, sizeof(v.ts)); buffer += sizeof(v.ts);
		*buffer++ = v.message.size();
		memcpy(buffer, v.message.data(), v.message.size()); buffer += v.message.size();
	}

	template<typename T> ArgType to_arg_type();
	template<> ArgType to_arg_type<uint8_t>()  { return ArgType::u8Type; }
	template<> ArgType to_arg_type<int8_t>()  { return ArgType::i8Type; }
	template<> ArgType to_arg_type<uint16_t>() { return ArgType::u16Type; }
	template<> ArgType to_arg_type<int16_t>() { return ArgType::i16Type; }
	template<> ArgType to_arg_type<uint32_t>() { return ArgType::u32Type; }
	template<> ArgType to_arg_type<int32_t>() { return ArgType::i32Type; }
	template<> ArgType to_arg_type<uint64_t>() { return ArgType::u64Type; }
	template<> ArgType to_arg_type<int64_t>() { return ArgType::i64Type; }
	template<> ArgType to_arg_type<float>()  { return ArgType::float32Type; }
	template<> ArgType to_arg_type<double>() { return ArgType::float64Type; }

	// TODO: make template is_numeric/float
	template<typename T>
	void operator()(const T& v)
	{
		const size_t NEEDED = 1 + sizeof(v);
		if (fail || buffer + NEEDED >= end)
		{
			fail = true; return;
		}
		buffer[0] = (char) to_arg_type<T>();
		memcpy(&buffer[1], &v, sizeof(v));
		buffer += NEEDED;
	}

	void operator()(const Field& f)
	{
		// NAME (without TYPE indicator)
		// value (with type indicator)
		const size_t NEEDED = f.name.size() + 1;
		if (fail || buffer + NEEDED >= end)
		{
			fail = true; return;
		}

		buffer[0] = (uint8_t)f.name.size();
		memcpy(&buffer[1], f.name.data(), f.name.size());
		buffer += NEEDED;

		std::visit(*this, f.value.data);
	}

	void operator()(const std::basic_string_view<char>& v)
	{
		const size_t NEEDED = v.size() + 2;
		if (fail || buffer + NEEDED >= end)
		{
			fail = true; return;
		}

		assert(v.size() < 256);
		buffer[0] = (char)ArgType::stringType;
		buffer[1] = (uint8_t)v.size();
		memcpy(&buffer[2], v.data(), v.size());
		buffer += NEEDED;
	}

	// string with length up to 255
	void operator()(const char* v, size_t N)
	{
		this->operator()(std::string_view(v, N));
	}

	void operator()(const char* v)
	{
		this->operator()(std::string_view(v));
	}
};

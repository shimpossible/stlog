#pragma once
#include <vector>
#include <string_view>
#include <variant>
#include <memory>

enum class ArgType
{
	u8Type,
	u16Type,
	u32Type,
	u64Type,
	i8Type,
	i16Type,
	i32Type,
	i64Type,
	float32Type,
	float64Type,
	stringType,
	objectType,
};

// like a Json object
// contains Fields, which are nammed ParamValue
class ObjectValue;

// A single value in an ObjectValue.  Basically a std::variant
class ParamValue
{
public:
	std::variant<
		int8_t,
		int16_t,
		int32_t,
		int64_t,
		uint8_t,                        // "foo" = 1234
		uint16_t,                       // "foo" = 1234
		uint32_t,                       // "foo" = 1234
		uint64_t,
		bool,                           // "foo" = false
		double,                         // "foo" = 1.23
		std::string_view,               // "foo" = "bar"
		std::shared_ptr<ObjectValue>,   // "foo" = {...}
		std::vector<ParamValue>			// "foo" = [1,2.3,false,'', {}]
	> data;

	~ParamValue()
	{
	}
	
	// move constructor
	ParamValue(ParamValue&& other) noexcept
	: data( std::move(other.data))
	{
	}

	// copy constructor
	ParamValue(const ParamValue& other) noexcept
	: data(std::move(other.data))
	{
	}
	
	ParamValue& operator=(const ParamValue& rhs) noexcept
	{
		data = rhs.data;
		return *this;
	}
	ParamValue& operator=(ParamValue&& rhs) noexcept
	{
		data = std::move(rhs.data);
		return *this;
	}

	template<typename T>
	ParamValue(T v) noexcept
	{
		data = v;
	}
	
	ParamValue(std::string_view v) noexcept
	{
		data = v;
	}
	ParamValue(const char* v) noexcept
	{
		data = std::string_view(v);
	}

	ParamValue(std::vector<ParamValue> v) noexcept
	{
		data = v;
	}

	// copy constructor
	ParamValue(const ObjectValue& v) noexcept
	{
		data = std::make_shared<ObjectValue>((v));
	}

	// move constructor..  this is the important one
	// as ObjectValues typically have alot of values with them
	// so a MOVE is much better than copying each
	ParamValue(ObjectValue&& v) noexcept;

};

// A Named ParamValue.  Basically a std::pair, but with constructor
// to make instantiation easier
struct Field
{
public:
	std::string_view name;
	ParamValue       value;

	~Field(){}

	Field()
	: value(0)
	{}

	// copy constructr
	Field(const Field& rhs)
		: name(rhs.name)
		, value(rhs.value)
	{
	}

	// move constructor
	Field(Field&& rhs) noexcept
		: name(rhs.name)
		, value(std::move(rhs.value))
	{
	}

	// copy assignment
	Field& operator=(const Field& rhs)
	{
		name  = rhs.name;
		value = rhs.value;
		return *this;
	}

	// move assignment
	Field& operator=(Field&& rhs) noexcept
	{
		name = std::move(rhs.name);
		value = std::move(rhs.value);

		return *this;
	}

	Field(const std::string_view& _name, const ParamValue& _value) noexcept
		: name(_name)
		, value(_value)
	{
	}

	Field(std::string_view&& _name, ParamValue&& _value) noexcept
		: name( std::move(_name))
		, value(std::move(_value))
	{
	}

	// need explicit ObjectValue parameter
	Field(const std::string_view& _name, const ObjectValue& _value) noexcept
		: name( _name )
		, value( _value )
	{
		// this is never called, but without in the RVAL constructor
		// isn't found
		fprintf(stderr, "field(Ref)\n");
	}

	Field(std::string_view&& _name, ObjectValue&& _value) noexcept
		: name(std::move(_name))
		, value(std::move(_value))
	{
		fprintf(stderr, "field(rval)\n");
	}
};

class ObjectValue
{
public:
	ObjectValue() {}
	~ObjectValue()
	{
	}

	typedef std::vector< Field>::iterator Iterator;
	std::vector< Field > data;

	void add(Field&& f)
	{
		data.push_back( std::move(f) );
	}

	// move constructor
	ObjectValue(ObjectValue&& other) noexcept
		: data(std::move(other.data))
	{
		fprintf(stderr, "obj(rval)\n");
		for (size_t i = 0; i < other.data.size(); i++)
		{
			fprintf(stderr, " %s \n", other.data[i].name.data());
		}

	}

	ObjectValue& operator=(ObjectValue&& other) noexcept
	{
		data = std::move(other.data);
		return *this;
	}


	// copy constructor
	ObjectValue(const ObjectValue& other) noexcept
		: data(other.data)
	{
		fprintf(stderr, "ref copy %d\n", other.data.size());
		for (size_t i = 0; i < other.data.size(); i++)
		{
			fprintf(stderr, " %s \n", other.data[i].name.data());
		}
		//data = other.data;
	}

	ObjectValue(const std::initializer_list<const Field> list)
		: data(list.begin(), list.end())
	{
	}
	ObjectValue(std::vector<Field>&& list)
		: data(std::move(list))
	{
	}
};

ParamValue::ParamValue(ObjectValue&& v) noexcept
{
	data = std::make_shared<ObjectValue>(std::move(v.data));
}

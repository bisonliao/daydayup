#ifndef __ANSI_STRING_H__
#define __ANSI_STRING_H__




#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

class AnsiString
{
public:
	AnsiString(const char * pstr);
	AnsiString(const AnsiString& s);
	AnsiString();
	~AnsiString();
	void concat(const char *pstr);
	void concat(const AnsiString & s);
	unsigned int length() const;
	AnsiString &operator=(const AnsiString& s);
	bool operator==(const AnsiString& s) const;
	char operator[](int index) const;	
	const char * c_str() const;
	AnsiString substring(unsigned int start, unsigned int len) const;
	void trimToSize();	/*缩小缓冲区，避免不必要的内存使用*/
	void clear();	/*变成一个空串*/
private:
	int m_bufsize;
	char * m_buffer;

private:
	void extend(int incr_size);
};

bool operator==(const char * pstr, const AnsiString & s);

#endif

// cstring.h : interface of the CString class
//
/////////////////////////////////////////////////////////////////////////////

#ifndef __CMBC_CSTRING_H__
#define __CMBC_CSTRING_H__

#include <stdarg.h>
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "basedef.h"
#include "winapi.h"

/////////////////////////////////////////////////////////////////////////////
// CString

#define EmptyString GetEmptyString()

class CString;

extern const CString& GetEmptyString();
extern BOOL IsValidString(LPCSTR lpsz, int nLength=-1);

#define strinc(p) (p+1)

struct CStringData
{
	long nRefs;             // reference count
	int nDataLength;        // length of data (including terminator)
	int nAllocLength;       // length of allocation
	// TCHAR data[nAllocLength]

	TCHAR* data()           // TCHAR* to managed data
		{ return (TCHAR*)(this+1); }
};

class CString
{
public:
// Constructors

	CString();							// constructs empty CString
	CString(const CString& stringSrc);	// copy constructor
	CString(TCHAR ch, int nRepeat = 1);	// from a single character
	CString(LPCSTR lpsz);				// from an ANSI string (converts to TCHAR)
	CString(LPCSTR lpch, int nLength);	// from a UNICODE string (converts to TCHAR)
	CString(const unsigned char* psz);	// from unsigned characters

// Attributes & Operations

	int GetLength() const;				// get data length
	BOOL IsEmpty() const;				// TRUE if zero length
	void Empty();						// clear contents to empty

	TCHAR GetAt(int nIndex) const;		// return single character at zero-based index
	TCHAR operator[](int nIndex) const;	// return single character at zero-based index
	void SetAt(int nIndex, TCHAR ch);	// set a single character at zero-based index
	operator LPCTSTR() const;			// return pointer to const string

	// overloaded assignment

	const CString& operator=(const CString& stringSrc);
	const CString& operator=(LPCSTR lpsz);
	const CString& operator=(TCHAR ch);

	// string concatenation

	const CString& operator+=(const CString& string);
	const CString& operator+=(LPCTSTR lpsz);
	const CString& operator+=(TCHAR ch);

	friend CString operator+(const CString& string1, const CString& string2);
	friend CString operator+(const CString& string, LPCTSTR lpsz);
	friend CString operator+(LPCTSTR lpsz, const CString& string);
	friend CString operator+(const CString& string, TCHAR ch);
	friend CString operator+(TCHAR ch, const CString& string);

	// string comparison

	int Compare(LPCTSTR lpsz) const;		// straight character comparison
	int CompareNoCase(LPCTSTR lpsz) const;	// compare ignoring case

	// simple sub-string extraction

	CString Mid(int nFirst, int nCount) const;	// return nCount characters starting at zero-based nFirst
	CString Mid(int nFirst) const;				// return all characters starting at zero-based nFirst
	CString Left(int nCount) const;				// return first nCount characters in string
	CString Right(int nCount) const;			// return nCount characters from end of string

	// upper/lower/reverse conversion

	void MakeUpper();						// NLS aware conversion to uppercase
	void MakeLower();						// NLS aware conversion to lowercase
	void MakeReverse();						// reverse string right-to-left

	// trimming whitespace (either side)

	void TrimRight();						// remove whitespace starting from right edge
	void TrimLeft();						// remove whitespace starting from left side

	// trimming anything (either side)

	void TrimRight(TCHAR chTarget);			// remove continuous occurrences of chTarget starting from right
	void TrimRight(LPCTSTR lpszTargets);	// remove continuous occcurrences of characters in passed string, starting from right
	void TrimLeft(TCHAR chTarget);			// remove continuous occurrences of chTarget starting from left
	void TrimLeft(LPCTSTR lpszTargets);		// remove continuous occcurrences of characters in passed string, starting from left

	// advanced manipulation
	
	int Replace(TCHAR chOld, TCHAR chNew);			// replace occurrences of chOld with chNew
	int Replace(LPCTSTR lpszOld, LPCTSTR lpszNew);	// replace occurrences of substring lpszOld with lpszNew;
													// empty lpszNew removes instances of lpszOld
	int Remove(TCHAR chRemove);						// remove occurrences of chRemove
	int Insert(int nIndex, TCHAR ch);				// insert character at zero-based index;
													// concatenates if index is past end of string
	int Insert(int nIndex, LPCTSTR pstr);			// insert substring at zero-based index; concatenates
													// if index is past end of string
	int Delete(int nIndex, int nCount = 1);			// delete nCount characters starting at zero-based index

	// searching

	
	int Find(TCHAR ch) const;					// find character starting at left, -1 if not found
	int ReverseFind(TCHAR ch) const;			// find character starting at right
	int Find(TCHAR ch, int nStart) const;		// find character starting at zero-based index and going right
	int FindOneOf(LPCTSTR lpszCharSet) const;	// find first instance of any character in passed string
	int Find(LPCTSTR lpszSub) const;			// find first instance of substring
	int Find(LPCTSTR lpszSub, int nStart) const;// find first instance of substring starting at zero-based index


	// simple formatting

	void Format(LPCTSTR lpszFormat, ...);				// printf-like formatting using passed string
	void FormatV(LPCTSTR lpszFormat, va_list argList);	// printf-like formatting using variable arguments parameter

	// Access to string implementation buffer as "C" character array

	LPTSTR GetBuffer(int nMinBufLength);		// get pointer to modifiable buffer at least as long as nMinBufLength
	void ReleaseBuffer(int nNewLength = -1);	// release buffer, setting length to nNewLength (or to first nul if -1)
	LPTSTR GetBufferSetLength(int nNewLength);	// get pointer to modifiable buffer exactly as long as nNewLength
	void FreeExtra();							// release memory allocated to but unused by string

	// Use LockBuffer/UnlockBuffer to turn refcounting off

	LPTSTR LockBuffer();			// turn refcounting back on
	void UnlockBuffer();			// turn refcounting off

// Implementation
public:
	~CString();
	int GetAllocLength() const;

protected:
	LPTSTR m_pchData;   // pointer to ref counted string data

	// implementation helpers
	CStringData* GetData() const;
	void Init();
	void AllocCopy(CString& dest, int nCopyLen, int nCopyIndex, int nExtraLen) const;
	void AllocBuffer(int nLen);
	void AssignCopy(int nSrcLen, LPCTSTR lpszSrcData);
	void ConcatCopy(int nSrc1Len, LPCTSTR lpszSrc1Data, int nSrc2Len, LPCTSTR lpszSrc2Data);
	void ConcatInPlace(int nSrcLen, LPCTSTR lpszSrcData);
	void CopyBeforeWrite();
	void AllocBeforeWrite(int nLen);
	void Release();
	static void Release(CStringData* pData);
	static int SafeStrlen(LPCTSTR lpsz);
	static void FreeData(CStringData* pData);
};

// CString
inline CStringData* CString::GetData() const
	{ return ((CStringData*)m_pchData)-1; }
inline void CString::Init()
	{ m_pchData = EmptyString.m_pchData; }
inline CString::CString(const unsigned char* lpsz)
	{ Init(); *this = (LPCSTR)lpsz; }

inline int CString::GetLength() const
	{ return GetData()->nDataLength; }
inline int CString::GetAllocLength() const
	{ return GetData()->nAllocLength; }
inline BOOL CString::IsEmpty() const
	{ return GetData()->nDataLength == 0; }
inline CString::operator LPCTSTR() const
	{ return m_pchData; }
inline int CString::SafeStrlen(LPCTSTR lpsz)
	{ return (lpsz == NULL) ? 0 : strlen(lpsz); }

// CString support (windows specific)
inline int CString::Compare(LPCTSTR lpsz) const
	{ ASSERT(IsValidString(lpsz)); return strcmp(m_pchData, lpsz); }    // MBCS/Unicode aware
inline int CString::CompareNoCase(LPCTSTR lpsz) const
	{ ASSERT(IsValidString(lpsz)); return strcasecmp(m_pchData, lpsz); }   // MBCS/Unicode aware

inline TCHAR CString::GetAt(int nIndex) const
{
	ASSERT(nIndex >= 0);
	ASSERT(nIndex < GetData()->nDataLength);
	return m_pchData[nIndex];
}
inline TCHAR CString::operator[](int nIndex) const
{
	// same as GetAt
	ASSERT(nIndex >= 0);
	ASSERT(nIndex < GetData()->nDataLength);
	return m_pchData[nIndex];
}
inline bool operator==(const CString& s1, const CString& s2)
	{ return s1.Compare(s2) == 0; }
inline bool operator==(const CString& s1, LPCTSTR s2)
	{ return s1.Compare(s2) == 0; }
inline bool operator==(LPCTSTR s1, const CString& s2)
	{ return s2.Compare(s1) == 0; }
inline bool operator!=(const CString& s1, const CString& s2)
	{ return s1.Compare(s2) != 0; }
inline bool operator!=(const CString& s1, LPCTSTR s2)
	{ return s1.Compare(s2) != 0; }
inline bool operator!=(LPCTSTR s1, const CString& s2)
	{ return s2.Compare(s1) != 0; }
inline bool operator<(const CString& s1, const CString& s2)
	{ return s1.Compare(s2) < 0; }
inline bool operator<(const CString& s1, LPCTSTR s2)
	{ return s1.Compare(s2) < 0; }
inline bool operator<(LPCTSTR s1, const CString& s2)
	{ return s2.Compare(s1) > 0; }
inline bool operator>(const CString& s1, const CString& s2)
	{ return s1.Compare(s2) > 0; }
inline bool operator>(const CString& s1, LPCTSTR s2)
	{ return s1.Compare(s2) > 0; }
inline bool operator>(LPCTSTR s1, const CString& s2)
	{ return s2.Compare(s1) < 0; }
inline bool operator<=(const CString& s1, const CString& s2)
	{ return s1.Compare(s2) <= 0; }
inline bool operator<=(const CString& s1, LPCTSTR s2)
	{ return s1.Compare(s2) <= 0; }
inline bool operator<=(LPCTSTR s1, const CString& s2)
	{ return s2.Compare(s1) >= 0; }
inline bool operator>=(const CString& s1, const CString& s2)
	{ return s1.Compare(s2) >= 0; }
inline bool operator>=(const CString& s1, LPCTSTR s2)
	{ return s1.Compare(s2) >= 0; }
inline bool operator>=(LPCTSTR s1, const CString& s2)
	{ return s2.Compare(s1) <= 0; }


/////////////////////////////////////////////////////////////////////////////
// Special implementations for CStrings
// it is faster to bit-wise copy a CString than to call an official
//   constructor - since an empty CString can be bit-wise copied

static inline void ConstructElement(CString* pNewData)
{
	memcpy(pNewData, &EmptyString, sizeof(CString));
}

static inline void DestructElement(CString* pOldData)
{
	pOldData->~CString();
}

static inline void CopyElement(CString* pSrc, CString* pDest)
{
	*pSrc = *pDest;
}

/////////////////////////////////////////////////////////////////////////////

#endif // __CMBC_CSTRING_H__

/////////////////////////////////////////////////////////////////////////////

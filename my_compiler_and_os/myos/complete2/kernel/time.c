#include "time.h"
#include "global.h"

#define MINUTE 60
#define HOUR (60*MINUTE)
#define DAY (24*HOUR)
#define YEAR (365*DAY)

#define CMOS_READ(addr) ({ \
		out_byte(0x70, 0x80|addr); \
		in_byte(0x71); \
		})

#define BCD_TO_BIN(val) ((val)=((val)&15) + ((val)>>4)*10)

void time_init(void)
{
	struct tm time;

	do {
		time.tm_sec = CMOS_READ(0);
		time.tm_min = CMOS_READ(2);
		time.tm_hour = CMOS_READ(4);
		time.tm_mday = CMOS_READ(7);
		time.tm_mon = CMOS_READ(8);
		time.tm_year = CMOS_READ(9);
	} while (time.tm_sec != CMOS_READ(0));
	BCD_TO_BIN(time.tm_sec);
	BCD_TO_BIN(time.tm_min);
	BCD_TO_BIN(time.tm_hour);
	BCD_TO_BIN(time.tm_mday);
	BCD_TO_BIN(time.tm_mon);
	BCD_TO_BIN(time.tm_year);
	time.tm_mon--;
	time.tm_year += 100; /*The number of years since 1900.*/
	g_startup_time = kernel_mktime(&time);
	g_uptime = 0;

	/* 1190191112 */
	/*
	   printk("%d-%d-%d %d:%d:%d, %u\n",
	   time.tm_year+1900,
	   time.tm_mon+1,
	   time.tm_mday,
	   time.tm_hour,
	   time.tm_min,
	   time.tm_sec,
	   g_startup_time);
	   while (1) {};
	   */
}   

/* interestingly, we assume leap-years */
static const unsigned int month[12] = {
	0,  
	DAY*(31),
	DAY*(31+29),
	DAY*(31+29+31),
	DAY*(31+29+31+30),
	DAY*(31+29+31+30+31),
	DAY*(31+29+31+30+31+30),
	DAY*(31+29+31+30+31+30+31),
	DAY*(31+29+31+30+31+30+31+31),
	DAY*(31+29+31+30+31+30+31+31+30),
	DAY*(31+29+31+30+31+30+31+31+30+31),
	DAY*(31+29+31+30+31+30+31+31+30+31+30)
};

uint32_t kernel_mktime(struct tm * tm)
{   
	uint32_t res;
	int year;

	year = tm->tm_year - 70;
	/* magic offsets (y+1) needed to get leapyears right.*/
	res = YEAR*year + DAY*((year+1)/4);
	res += month[tm->tm_mon];
	/* and (y+2) here. If it wasn't a leap-year, we have to adjust */
	if (tm->tm_mon>1 && ((year+2)%4))
		res -= DAY;
	res += DAY*(tm->tm_mday-1);
	res += HOUR*tm->tm_hour;
	res += MINUTE*tm->tm_min;
	res += tm->tm_sec;
	return res;
}
void localtime(uint32_t sec, struct tm * t)
{
	int year, i, mon, leap;
	if (t == NULL)
	{
		return;
	}
	memset(t, 0, sizeof(struct tm));
	year = sec / YEAR;
	t->tm_year = year + 70;
	sec -= year*YEAR;
	leap = DAY*((year+1)/4);
	if (sec >= leap)
	{
		sec -= leap;
	}
	for (i = 0; i < 12; ++i)
	{
		if (sec >= month[i])
		{
		}
		else
		{
			break;
		}
	}
	if (i >= 12 || i == 0)
		panic("localtime() BUG!");

	t->tm_mon = i;
	sec -= month[i-1];


	t->tm_mday = sec /DAY + 1;
	sec -= (t->tm_mday-1) * DAY;

	t->tm_hour = sec / HOUR;
	sec -= t->tm_hour*HOUR;

	t->tm_min = sec /MINUTE;
	sec -= t->tm_min  * MINUTE;
	t->tm_sec = sec;
	return;
}
uint32_t current_time()
{
	/*
	uint64_t res;
	div_uint64(g_ticks, HZ, &res);
	return g_startup_time + res & 0xffffffff;
	*/
	return g_startup_time+g_uptime;
}
uint32_t up_time()
{
	/*
	uint64_t res;
	div_uint64(g_ticks, HZ, &res);
	return res & 0xffffffff;
	*/
	return g_uptime;
}

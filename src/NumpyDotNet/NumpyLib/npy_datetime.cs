/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2018-2019
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using npy_datetime = System.Int64;
using npy_longlong = System.Int64;
using npy_int64 = System.Int64;
using npy_timedelta = System.Int64;
using System.Diagnostics;

namespace NumpyLib
{
    internal partial class numpyinternal
    {
        /* Offset for number of days between Dec 31, 1969 and Jan 1, 0001
         * Assuming Gregorian calendar was always in effect (proleptic Gregorian
         * calendar)
         */

        /* Calendar Structure for Parsing Long -> Date */
        struct ymdstruct
        {
            public int year, month, day;
        };

        struct hmsstruct
        {
            public int hour, min, sec;
        };


        /*
         * Create a datetime value from a filled datetime struct and resolution unit.
         */
        internal static npy_datetime NpyArray_DatetimeStructToDatetime(NPY_DATETIMEUNIT fr, npy_datetimestruct d)
        {
            npy_datetime ret;

            if (fr == NPY_DATETIMEUNIT.NPY_FR_Y)
            {
                ret = d.year - 1970;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_M)
            {
                ret = (d.year - 1970) * 12 + d.month - 1;
            }
            else
            {
                npy_longlong days; /* The absolute number of days since Jan 1, 1970 */

                days = days_from_ymd(d.year, d.month, d.day);
                if (fr == NPY_DATETIMEUNIT.NPY_FR_W)
                {
                    /* This is just 7-days for now. */
                    if (days >= 0)
                    {
                        ret = days / 7;
                    }
                    else
                    {
                        ret = (days - 6) / 7;
                    }
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_B)
                {
                    npy_longlong x;
                    int dotw = day_of_week(days);

                    if (dotw > 4)
                    {
                        /* Invalid business day */
                        ret = 0;
                    }
                    else
                    {
                        if (days >= 0)
                        {
                            /* offset to adjust first week */
                            x = days - 4;
                        }
                        else
                        {
                            x = days - 2;
                        }
                        ret = 2 + (x / 7) * 5 + x % 7;
                    }
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_D)
                {
                    ret = days;
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_h)
                {
                    ret = days * 24 + d.hour;
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_m)
                {
                    ret = days * 1440 + d.hour * 60 + d.min;
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_s)
                {
                    ret = days * (npy_int64)(86400) + secs_from_hms(d.hour, d.min, d.sec, 1);
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_ms)
                {
                    ret = days * (npy_int64)(86400000) + secs_from_hms(d.hour, d.min, d.sec, 1000)
                        + (d.us / 1000);
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_us)
                {
                    npy_int64 num = 86400 * 1000;
                    num *= (npy_int64)(1000);
                    ret = days * num + secs_from_hms(d.hour, d.min, d.sec, 1000000) + d.us;
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_ns)
                {
                    npy_int64 num = 86400 * 1000;
                    num *= (npy_int64)(1000 * 1000);
                    ret = days * num + secs_from_hms(d.hour, d.min, d.sec, 1000000000) + d.us * (npy_int64)(1000) + (d.ps / 1000);
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_ps)
                {
                    npy_int64 num2 = 1000 * 1000;
                    npy_int64 num1;

                    num2 *= (npy_int64)(1000 * 1000);
                    num1 = (npy_int64)(86400) * num2;
                    ret = days * num1 + secs_from_hms(d.hour, d.min, d.sec, (int)num2) + d.us * (npy_int64)(1000000) + d.ps;
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_fs)
                {
                    /* only 2.6 hours */
                    npy_int64 num2 = 1000000;
                    num2 *= (npy_int64)(1000000);
                    num2 *= (npy_int64)(1000);

                    /* get number of seconds as a postive or negative number */
                    if (days >= 0)
                    {
                        ret = secs_from_hms(d.hour, d.min, d.sec, 1);
                    }
                    else
                    {
                        ret = ((d.hour - 24) * 3600 + d.min * 60 + d.sec);
                    }
                    ret = ret * num2 + d.us * (npy_int64)(1000000000)
                        + d.ps * (npy_int64)(1000) + (d.as1 / 1000);
                }
                else if (fr == NPY_DATETIMEUNIT.NPY_FR_as)
                {
                    /* only 9.2 secs */
                    npy_int64 num1, num2;

                    num1 = 1000000;
                    num1 *= (npy_int64)(1000000);
                    num2 = num1 * (npy_int64)(1000000);

                    if (days >= 0)
                    {
                        ret = d.sec;
                    }
                    else
                    {
                        ret = d.sec - 60;
                    }
                    ret = ret * num2 + d.us * num1 + d.ps * (npy_int64)(1000000) + d.as1;
                }
                else
                {
                    /* Shouldn't get here */
                    NpyErr_SetString(npyexc_type.NpyExc_ValueError, "invalid internal frequency");
                    ret = -1;
                }
            }

            return ret;
        }

        /*
         * Fill the datetime struct from the value and resolution unit.
         */
        internal static void  NpyArray_DatetimeToDatetimeStruct(npy_datetime val, NPY_DATETIMEUNIT fr, npy_datetimestruct result)
        {
            int year = 1970;
            int month = 1, day = 1,
            hour = 0, min = 0, sec = 0,
            us = 0, ps = 0, as1 = 0;

            npy_int64 tmp;
            ymdstruct ymd;
            hmsstruct hms;

            /*
             * Note that what looks like val / N and val % N for positive numbers maps to
             * [val - (N-1)] / N and [N-1 + (val+1) % N] for negative numbers (with the 2nd
             * value, the remainder, being positive in both cases).
             */
            if (fr == NPY_DATETIMEUNIT.NPY_FR_Y)
            {
                year = (int)(1970 + val);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_M)
            {
                if (val >= 0)
                {
                    year = (int)(1970 + val / 12);
                    month = (int)(val % 12 + 1);
                }
                else
                {
                    year = (int)(1969 + (val + 1) / 12);
                    month = (int)(12 + (val + 1) % 12);
                }
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_W)
            {
                /* A week is the same as 7 days */
                ymd = days_to_ymdstruct(val * 7);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_B)
            {
                /* Number of business days since Thursday, 1-1-70 */
                npy_longlong absdays;
                /*
                 * A buisness day is M T W Th F (i.e. all but Sat and Sun.)
                 * Convert the business day to the number of actual days.
                 *
                 * Must convert [0,1,2,3,4,5,6,7,...] to
                 *                  [0,1,4,5,6,7,8,11,...]
                 * and  [...,-9,-8,-7,-6,-5,-4,-3,-2,-1,0] to
                 *        [...,-13,-10,-9,-8,-7,-6,-3,-2,-1,0]
                 */
                if (val >= 0)
                {
                    absdays = 7 * ((val + 3) / 5) + ((val + 3) % 5) - 3;
                }
                else
                {
                    /* Recall how C computes / and % with negative numbers */
                    absdays = 7 * ((val - 1) / 5) + ((val - 1) % 5) + 1;
                }
                ymd = days_to_ymdstruct(absdays);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_D)
            {
                ymd = days_to_ymdstruct(val);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_h)
            {
                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / 24);
                    hour = (int)(val % 24);
                }
                else
                {
                    ymd = days_to_ymdstruct((val - 23) / 24);
                    hour = (int)(23 + (val + 1) % 24);
                }
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_m)
            {
                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / 1440);
                    min = (int)(val % 1440);
                }
                else
                {
                    ymd = days_to_ymdstruct((val - 1439) / 1440);
                    min = (int)(1439 + (val + 1) % 1440);
                }
                hms = seconds_to_hmsstruct(min * 60);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
                hour = hms.hour;
                min = hms.min;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_s)
            {
                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / 86400);
                    sec = (int)(val % 86400);
                }
                else
                {
                    ymd = days_to_ymdstruct((val - 86399) / 86400);
                    sec = (int)(86399 + (val + 1) % 86400);
                }
                hms = seconds_to_hmsstruct(val);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
                hour = hms.hour;
                min = hms.min;
                sec = hms.sec;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_ms)
            {
                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / 86400000);
                    tmp = val % 86400000;
                }
                else
                {
                    ymd = days_to_ymdstruct((val - 86399999) / 86400000);
                    tmp = 86399999 + (val + 1) % 86399999;
                }
                hms = seconds_to_hmsstruct(tmp / 1000);
                us = (int)((tmp % 1000) * 1000);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
                hour = hms.hour;
                min = hms.min;
                sec = hms.sec;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_us)
            {
                npy_int64 num1, num2;
                num1 = 86400000;
                num1 *= 1000;
                num2 = num1 - 1;
                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / num1);
                    tmp = val % num1;
                }
                else
                {
                    ymd = days_to_ymdstruct((val - num2) / num1);
                    tmp = num2 + (val + 1) % num1;
                }
                hms = seconds_to_hmsstruct(tmp / 1000000);
                us = (int)(tmp % 1000000);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
                hour = hms.hour;
                min = hms.min;
                sec = hms.sec;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_ns)
            {
                npy_int64 num1, num2, num3;
                num1 = 86400000;
                num1 *= 1000000000;
                num2 = num1 - 1;
                num3 = 1000000;
                num3 *= 1000000;
                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / num1);
                    tmp = val % num1;
                }
                else
                {
                    ymd = days_to_ymdstruct((val - num2) / num1);
                    tmp = num2 + (val + 1) % num1;
                }
                hms = seconds_to_hmsstruct(tmp / 1000000000);
                tmp = tmp % 1000000000;
                us = (int)(tmp / 1000);
                ps = (int)((tmp % 1000) * (npy_int64)(1000));
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
                hour = hms.hour;
                min = hms.min;
                sec = hms.sec;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_ps)
            {
                npy_int64 num1, num2, num3;
                num3 = 1000000000;
                num3 *= (npy_int64)(1000);
                num1 = (npy_int64)(86400) * num3;
                num2 = num1 - 1;

                if (val >= 0)
                {
                    ymd = days_to_ymdstruct(val / num1);
                    tmp = val % num1;
                }
                else
                {
                    ymd = days_to_ymdstruct((val - num2) / num1);
                    tmp = num2 + (val + 1) % num1;
                }
                hms = seconds_to_hmsstruct(tmp / num3);
                tmp = tmp % num3;
                us = (int)(tmp / 1000000);
                ps = (int)(tmp % 1000000);
                year = ymd.year;
                month = ymd.month;
                day = ymd.day;
                hour = hms.hour;
                min = hms.min;
                sec = hms.sec;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_fs)
            {
                /* entire range is only += 2.6 hours */
                npy_int64 num1, num2;
                num1 = 1000000000;
                num1 *= (npy_int64)(1000);
                num2 = num1 * (npy_int64)(1000);

                if (val >= 0)
                {
                    sec = (int)(val / num2);
                    tmp = val % num2;
                    hms = seconds_to_hmsstruct(sec);
                    hour = hms.hour;
                    min = hms.min;
                    sec = hms.sec;
                }
                else
                {
                    /* tmp (number of fs) will be positive after this segment */
                    year = 1969;
                    day = 31;
                    month = 12;
                    sec = (int)((val - (num2 - 1)) / num2);
                    tmp = (num2 - 1) + (val + 1) % num2;
                    if (sec == 0)
                    {
                        /* we are at the last second */
                        hour = 23;
                        min = 59;
                        sec = 59;
                    }
                    else
                    {
                        hour = 24 + (sec - 3599) / 3600;
                        sec = 3599 + (sec + 1) % 3600;
                        min = sec / 60;
                        sec = sec % 60;
                    }
                }
                us = (int)(tmp / 1000000000);
                tmp = tmp % 1000000000;
                ps = (int)(tmp / 1000);
                as1 = (int)((tmp % 1000) * (npy_int64)(1000));
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_as)
            {
                /* entire range is only += 9.2 seconds */
                npy_int64 num1, num2, num3;
                num1 = 1000000;
                num2 = num1 * (npy_int64)(1000000);
                num3 = num2 * (npy_int64)(1000000);
                if (val >= 0)
                {
                    hour = 0;
                    min = 0;
                    sec = (int)(val / num3);
                    tmp = val % num3;
                }
                else
                {
                    year = 1969;
                    day = 31;
                    month = 12;
                    hour = 23;
                    min = 59;
                    sec = (int)(60 + (val - (num3 - 1)) / num3);
                    tmp = (num3 - 1) + (val + 1) % num3;
                }
                us = (int)(tmp / num2);
                tmp = tmp % num2;
                ps = (int)(tmp / num1);
                as1 = (int)(tmp % num1);
            }
            else
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError, "invalid internal time resolution");
            }

            result.year = year;
            result.month = month;
            result.day = day;
            result.hour = hour;
            result.min = min;
            result.sec = sec;
            result.us = us;
            result.ps = ps;
            result.as1 = as1;

            return;
        }

        /*
         * Fill the timedelta struct from the timedelta value and resolution unit.
         */
        internal static void NpyArray_TimedeltaToTimedeltaStruct(npy_timedelta val, NPY_DATETIMEUNIT fr, npy_timedeltastruct result)
        {
            npy_longlong day = 0;
            int sec = 0, us = 0, ps = 0, as1= 0;
            bool negative = false;

            /*
             * Note that what looks like val / N and val % N for positive numbers maps to
             * [val - (N-1)] / N and [N-1 + (val+1) % N] for negative numbers (with
             * the 2nd value, the remainder, being positive in both cases).
             */

            if (val < 0)
            {
                val = -val;
                negative = true;
            }
            if (fr == NPY_DATETIMEUNIT.NPY_FR_Y)
            {
                day = (long)(val * _DAYS_PER_YEAR);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_M)
            {
                day = (long)(val * _DAYS_PER_MONTH);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_W)
            {
                day = val * 7;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_B)
            {
                /* Number of business days since Thursday, 1-1-70 */
                day = (val * 7) / 5;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_D)
            {
                day = val;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_h)
            {
                day = val / 24;
                sec = (int)((val % 24) * 3600);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_m)
            {
                day = val / 1440;
                sec = (int)((val % 1440) * 60);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_s)
            {
                day = val / (86400);
                sec = (int)(val % 86400);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_ms)
            {
                day = val / 86400000;
                val = val % 86400000;
                sec = (int)(val / 1000);
                us = (int)((val % 1000) * 1000);
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_us)
            {
                npy_int64 num1;
                num1 = 86400000;
                num1 *= 1000;
                day = val / num1;
                us = (int)(val % num1);
                sec = us / 1000000;
                us = us % 1000000;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_ns)
            {
                npy_int64 num1;
                num1 = 86400000;
                num1 *= 1000000;
                day = val / num1;
                val = val % num1;
                sec = (int)(val / 1000000000);
                val = val % 1000000000;
                us = (int)(val / 1000);
                ps = (int)((val % 1000) * (npy_int64)(1000));
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_ps)
            {
                npy_int64 num1, num2;
                num2 = 1000000000;
                num2 *= (npy_int64)(1000);
                num1 = (npy_int64)(86400) * num2;

                day = val / num1;
                ps = (int)(val % num1);
                sec = (int)(ps / num2);
                ps = (int)(ps % num2);
                us = ps / 1000000;
                ps = ps % 1000000;
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_fs)
            {
                /* entire range is only += 9.2 hours */
                npy_int64 num1, num2;
                num1 = 1000000000;
                num2 = num1 * (npy_int64)(1000000);

                day = 0;
                sec = (int)(val / num2);
                val = val % num2;
                us = (int)(val / num1);
                val = val % num1;
                ps = (int)(val / 1000);
                as1 = (int)((val % 1000) * (npy_int64)(1000));
            }
            else if (fr == NPY_DATETIMEUNIT.NPY_FR_as)
            {
                /* entire range is only += 2.6 seconds */
                npy_int64 num1, num2, num3;
                num1 = 1000000;
                num2 = num1 * (npy_int64)(1000000);
                num3 = num2 * (npy_int64)(1000000);
                day = 0;
                sec = (int)(val / num3);
                as1 = (int)(val % num3);
                us = (int)(as1 / num2);
                as1 = (int)(as1 % num2);
                ps = (int)(as1 / num1);
                as1 = (int)(as1 % num1);
            }
            else
            {
                NpyErr_SetString(npyexc_type.NpyExc_RuntimeError,
                                 "invalid internal time resolution");
            }

            if (negative)
            {
                result.day = -day;
                result.sec = -sec;
                result.us = -us;
                result.ps = -ps;
                result.as1 = -as1;
            }
            else
            {
                result.day = day;
                result.sec = sec;
                result.us = us;
                result.ps = ps;
                result.as1    = as1;
            }
            return;
        }

        /*
         * Modified version of mxDateTime function
         * Returns absolute number of days since Jan 1, 1970
         * assuming a proleptic Gregorian Calendar
         * Raises a ValueError if out of range month or day
         * day -1 is Dec 31, 1969, day 0 is Jan 1, 1970, day 1 is Jan 2, 1970
         */
        static npy_longlong days_from_ymd(int year, int month, int day)
        {

            /* Calculate the absolute date */
            int leap;
            npy_longlong yearoffset, absdate;

            /* Is it a leap year ? */
            leap = is_leapyear(year);

            /* Negative month values indicate months relative to the years end */
            if (month < 0) month += 13;
            Debug.Assert(month >= 1 && month <= 12);

            /* Negative values indicate days relative to the months end */
            if (day < 0) day += days_in_month[leap][month - 1] + 1;
            Debug.Assert(day >= 1 && day <= days_in_month[leap][month - 1]);

            /*
             * Number of days between Dec 31, (year - 1) and Dec 31, 1969
             *    (can be negative).
             */
            yearoffset = year_offset(year);

            if (NpyErr_Occurred())
            {
                goto onError;
            }

            /*
             * Calculate the number of days using yearoffset
             * Jan 1, 1970 is day 0 and thus Dec. 31, 1969 is day -1
             */
            absdate = day - 1 + month_offset[leap][month - 1] + yearoffset;

            return absdate;

            onError:
            return 0;

        }
        /*
         * Return the day of the week for the given absolute date.
         * Monday is 0 and Sunday is 6
         */
        static int day_of_week(npy_longlong absdate)
        {
            /* Add in four for the Thursday on Jan 1, 1970 (epoch offset)*/
            absdate += 4;

            if (absdate >= 0)
            {
                return (int)(absdate % 7);
            }
            else
            {
                return (int)(6 + (absdate + 1) % 7);
            }
        }

        /*
         * Takes a number of days since Jan 1, 1970 (positive or negative)
         * and returns the year. month, and day in the proleptic
         * Gregorian calendar
         *
         * Examples:
         *
         * -1 returns 1969, 12, 31
         * 0  returns 1970, 1, 1
         * 1  returns 1970, 1, 2
         */

        static ymdstruct days_to_ymdstruct(npy_datetime dlong)
        {
            ymdstruct ymd;
            long year;
            npy_longlong yearoffset;
            int leap;
            int dayoffset;
            int month = 1, day = 1;
            int[] monthoffset;

            dlong += 1;

            /* Approximate year */
            year = (long)(1970 + dlong / 365.2425);

            /* Apply corrections to reach the correct year */
            while (true)
            {
                /* Calculate the year offset */
                yearoffset = year_offset(year);

                /*
                 * Backward correction: absdate must be greater than the
                 * yearoffset
                 */
                if (yearoffset >= dlong)
                {
                    year--;
                    continue;
                }

                dayoffset = (int)(dlong - yearoffset);
                leap = is_leapyear(year);

                /* Forward correction: non leap years only have 365 days */
                if (dayoffset > 365 && leap == 0)
                {
                    year++;
                    continue;
                }
                break;
            }

            /* Now iterate to find the month */
            monthoffset = month_offset[leap];
            for (month = 1; month < 13; month++)
            {
                if (monthoffset[month] >= dayoffset)
                    break;
            }
            day = dayoffset - month_offset[leap][month - 1];

            ymd.year = (int)year;
            ymd.month = month;
            ymd.day = day;

            return ymd;
        }

        /*
         * Converts an integer number of seconds in a day to hours minutes seconds.
         * It assumes seconds is between 0 and 86399.
         */

        static hmsstruct seconds_to_hmsstruct(npy_longlong dlong)
        {
            long hour, minute, second;
            hmsstruct hms;

            hour = dlong / 3600;
            minute = (dlong % 3600) / 60;
            second = dlong - (hour * 3600 + minute * 60);

            hms.hour = (int)hour;
            hms.min = (int)minute;
            hms.sec = (int)second;

            return hms;
        }

        static npy_longlong year_offset(npy_longlong year)
        {
            /* Note that 477 == 1969/4 - 1969/100 + 1969/400 */
            year--;
            if (year >= 0 || -1 / 4 == -1)
                return (year - 1969) * 365 + year / 4 - year / 100 + year / 400 - 477;
            else
                return (year - 1969) * 365 + (year - 3) / 4 - (year - 99) / 100 + (year - 399) / 400 - 477;
        }

        /* Returns absolute seconds from an hour, minute, and second */
        internal static long secs_from_hms(int hour, int min, int sec, int multiplier)
        {
            return ((hour) * 3600 + (min) * 60 + (sec)) * (npy_int64)(multiplier);
        }

        /* Uses Average values when frequency is Y, M, or B */

        const double _DAYS_PER_MONTH = 30.436875;
        const double _DAYS_PER_YEAR = 365.2425;

        /* Table with day offsets for each month (0-based, without and with leap) */
        static int[] month_row1 = new int[] { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365 };
        static int[] month_row2 = new int[] { 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366 };
        static int[][] month_offset = { month_row1, month_row2 };

        /* Table of number of days in a month (0-based, without and with leap) */
        static int[] days_in_month1 = new int[] { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        static int[] days_in_month2 = new int[] { 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        static int[][] days_in_month = { days_in_month1, days_in_month2 };

        /* Return 1/0 iff year points to a leap year in calendar. */
        static int is_leapyear(long year)
        {
            bool b = (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
            return b ? 1 : 0;
        }


    }
}

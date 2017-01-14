
OpenXC (OXC) Time Series Objects (TSO) and Collections (TSC) in Python. 
=================================================================================

Description
-------------

[OpenXC](http://openxcplatform.com/) is a vehicle data collection tool/platform that
utilizes the [OBD-II port](https://en.wikipedia.org/wiki/On-board_diagnostics) required 
in new vehicles since 1996. Depending on where you are in the data stream, you can 
get protobuf signals, JSON signals, or JSON dumps of signals. By "signal" here I 
mean something of the form (device id,time,data name and value); reviewing the 
[OpenXC github repo](https://github.com/openxc) is a good way to learn more 
about the actual signal formatting in JSON. 

I've found that this format is difficult to handle as an analyst, preferring time
series models of the data. It isn't too hard to "pivot" the stream to create a 
time-indexed table (ie [DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)) 
of signals, but such tables have many empty cells. This can probably be handled with "NaN-concious" 
routines in pandas/numpy, including sparse Series/DataFrames, so some of this project may end up just wrapping that
functionality for OXC specific purposes. 

Run tests with

    >>> python tests/oxctsd_utests.py

(or python3, depending on your environment settings). I haven't tested with python 2.7+, only 3.5. 

Status
-------------
This is preliminary work I'm being proactive about publishing, and has
no warranty. 

License
-------------
This code is openly available for use under an Apache 2.0+ license for non-commercial
purposes. If you're interested in using this code for commercial purposes, related to
vehicles or not, contact me at morrowwr@gmail.com. 

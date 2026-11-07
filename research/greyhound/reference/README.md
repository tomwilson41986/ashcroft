# Greyhound reference tables

`gb_tracks.csv`: every track name in GBGB's results from 1 Jan 2014 to 5 Nov 2026 (33 names, read off the GBGB API day
by day; 122 days hit GBGB's 500-race list cap, so a meeting on those days may be missing). Compiled 6 Nov 2026 from
public pages; every value carries the URL it came from, and a value not found in a source is blank, never estimated.

- coordinates: all 33, from fixed-revision Wikipedia links (used by sources/noaa_isd.py to pick each track's weather
  stations);
- circumference: 22 of 33; blank where none was found (Belle Vue, Dunstall Park, Hall Green, Peterborough, Poole,
  Valley, The Valley) or the sources disagree (Doncaster, Oxford, Swindon, Suffolk Downs: every candidate is in notes);
- distances raced: from the GBGB data (30 races or more), so old and new layouts can mix (Towcester, Swindon);
- the run to the first bend: 15 tracks with a figure, most from bettingodds.com pages, which are undated and some
  plainly old (Central Park's 443 m predates its 2023 rebuild): treat as possibly out of date;
- `betfair_name` is empty: Betfair's names could not be checked from the compiling machine;
- uncertain values (Henlow's coordinate taken from the article's title coordinate, as the infobox repeats Harlow's;
  Nottingham's to 3 decimals; Crayford's first-bend figure; Oxford and Suffolk Downs marked suspended after SIS ended
  its contract from 6 Apr 2026) are noted row by row.

Renames: Sittingbourne became Central Park (June 2016); Mildenhall became Suffolk Downs; Pelaw Grange became Star
Pelaw (GBGB's own name from Dec 2024); The Valley is Valley. Hove is Brighton & Hove; Monmore is Wolverhampton.

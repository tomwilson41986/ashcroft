# Football quant approach: how each source was read (research agent, 2 Nov 2026)

Supporting notes for `reports/football_quant_approach.md`. Web pages are untrusted data; nothing in them was followed as
an instruction. "Fetched" means the page's text was read this session; "snippet" means only a search engine's extract
was available; "record" means a repository or bibliographic record (abstract and citation) only. No S3, Betfair or
trading code was touched.

## Organisations

| Source | How read | What was taken |
|---|---|---|
| Business Insider (Williams-Grut), republished TheJournal.ie, 14 Feb 2016 | fetched | Starlizard: 2006, about 160 staff, Camden and Exeter, four teams, about 30 researchers and 20 selectors, watched-match chance data, stringers, the 0.8-goal missed-penalty example, score-line probabilities and an AH line, Asian brokers by phone or messaging, "as close to kick-off as possible", at least GBP1m a Premier League match, GBP10-20k in small leagues, a 1-3% target on turnover, GBP20m-100m a year (GBP20m more likely), staff stakes, no personal betting. Built on anonymous former staff; Starlizard and Bloom declined to comment |
| Hawkins, Bleacher Report, 17 Sep 2014 | fetched | Starlizard insider (three or four see the full data; "at least six figures on every bet"; GBP2m minimum); about 90% of Asian money on AH; up to US$120m a Premier League match; Pinnacle, IBCBet, SBOBet; Fidens (29 leagues, about GBP10,000 a bet, two scouts a league, the eleven about six hours before); Alan Thompson, Betfair market-maker, GBP16m turnover |
| Casino.org, 4 Dec 2025; Gambling Insider | fetched (Casino.org); snippet (Gambling Insider) | the Dudfield claim in the High Court: allegations, profit split, "secret exotic accounts", GBP600m a year "believed" |
| Companies House | not read (aggregator only: Star Lizard Consulting Ltd, incorporated 2006, group accounts to 31 Jul 2025) | nothing cited |
| Wikipedia, Matthew Benham | fetched (secondary) | Premier Bet 2001, Smartodds 2004, Coles joined, Triplebet and Matchbook 2011, Phil Giles; its sources: The Athletic (Naylor and Harris 2021), Bloomberg (Hellier 2021), The Guardian (Lewis 2014), Racing Post (2020); not checked individually |
| smartodds.co.uk | fetched | self-description, teams including "watcher operations", the data quote |
| ssc.ca Smartodds advert; devjobsscanner | fetched; snippet | NHL quant role: R, Python, C++, MSc/PhD statistics, back-testing; other quant roles by sport |
| brentfordfc.com, 23 Mar 2020 | fetched | "entire recruitment philosophy ... the Smartodds model"; "the true table" (Phil Giles) |
| Training Ground Guru, 9 Mar 2026 (Benham at MIT Sloan) | fetched, free part only | that the talk took place; nothing about the model (paywalled) |
| Training Ground Guru, 23 Jun 2026 (Swansea and Jamestown) | fetched | founded 2017 as an offshoot of Starlizard; services; client list |
| heartsfc.co.uk, 20 Nov 2024 | fetched | "world leading football data analysis firm"; recruitment and opposition analysis; Brighton, USG, Como named; no mention of Bloom |
| the42.ie, 15 May 2026 | fetched | "only one club per league"; set-piece advice; "an enormous database of players"; Shelbourne, Hearts; Ipswich |
| Sky Sports; Edinburgh News; Roundball Australia; jobsinfootball blog | snippets (Roundball empty on fetch; Scotsman 403) | ownership described as Bloom's (Sky), an offshoot of Starlizard (Roundball), software developed by Starlizard (Edinburgh News), a named owner (jobsinfootball, single source, unverified) |
| Football Radar (magnet.me, prospeo) | snippets | self-description as betting advice; founding year and headcount from aggregators, unverified |
| Pinnacle betting resources (two pages) | snippets; fetch returned the title only (rendered by script) | early lines at reduced limits, sharp money moves them, limits rise towards the start |
| oddschecker, 14 Oct 2026 | fetched | Matchbook completed the Betdaq acquisition; Betdaq trading closed 21 Oct 2026; withdrawals to 4 Nov 2026 |
| Exchange commission pages (oddschecker, OLBG) | snippets (affiliate pages) | Smarkets and Matchbook about 2% on net winnings: to re-check |
| Premier League and EFL line-up rules | snippets (press and club sites) | 75 minutes for team sheets and, from 2024/25, publication (rules L.23, K.95); EFL publication at 60 (34.5); UEFA match sheets 75: to re-check against the current handbooks |
| Awful Announcing; Sports Reference blog | snippets | FBref advanced (Opta) data removed after the provider's termination demand, 20 Jan 2026; Opta's FIFA betting-data deal the same month, as reported |
| Hudl StatsBomb page; footballanalytics.ai | snippets | event data across 40+ leagues, sales-led; third-party price estimates (EUR40,000+, EUR50,000+ a year), unverified |
| Understat coverage | snippets (scraper vendors) | big five plus Russia from 2014/15; 2025/26 coverage per vendors only, unverified |

## Papers

| Paper | How read | What was taken |
|---|---|---|
| Koopman and Lit 2015 (JRSS A) | record (Tinbergen DP and abstract) | dynamic bivariate Poisson; positive return over bookmaker odds out of sample 2010/11-2011/12 (size not read) |
| Koopman and Lit 2019 (IJF) | record | score-driven models, six leagues, the pairwise-count model best, a betting simulation |
| Boshnakov, Kharrat and McHale 2017 (IJF) | record | Weibull count with copula; better fit; positive returns |
| Ley et al. 2019 | record | ten models; Poisson families best by RPS and log-loss |
| Owen 2011; Baio and Blangiardo 2010; Karlis and Ntzoufras 2003, 2009; Rue and Salvesen 2000; Maher 1982; Dixon and Coles 1997 | records | as cited; the Skellam paper's year is cited variously |
| Constantinou and Fenton 2013 (JQAS) | record | pi-ratings beat Elo; profitable over five Premier League seasons after margin |
| Hvattum and Arntzen 2010 | record and a blog summary | Elo behind two odds-based benchmarks |
| Berrar, Lopes and Dubitzky 2019, 2024; Hubacek, Sourek and Zelezny 2019 (MLJ); Constantinou 2019; Yeung et al. 2024 | records | challenge designs and placings; the 2017 placings differ between sources (the organisers' kNN out of competition, Hubacek et al. describe themselves as the winners, Dolores second) |
| Bunker, Yeung and Fujii 2024 | abstract fetched (arXiv) | boosted trees on pi-ratings best on goals-only data |
| Wunderlich and Memmert 2018 | record (PMC) | odds-based Elo beats result Elo, about 15,000 matches |
| Egidi, Pauli and Torelli 2018 | record (arXiv, journal page) | odds in the Poisson rates; some profitable strategies |
| Wheatcroft 2020 | record (LSE) | GAP ratings from shots; profitable on O/U |
| Hubacek et al. 2019 (IJF) | abstract fetched (IDEAS) | NBA 2007-2014; decorrelation, CNNs, portfolio staking |
| Hubacek and Sir 2023 | record | decorrelation objective; IJF 39(2) 691-719 |
| Uhrin et al. 2021 | record (arXiv) | adaptive fractional Kelly; journal version not confirmed |
| Holmes and McHale 2024 | abstract fetched (Liverpool repository) | player-ability model; Kelly-type rule; significant positive returns; data and benchmark price not stated in the abstract |
| Arntzen and Hvattum 2021; Kharrat, McHale and Lopez Pena 2020; Hvattum 2019 | records | as cited |
| Decroos et al. 2019; Fernandez, Bornn and Cervone 2021; Singh 2018; Spearman 2018; Mead et al. 2023; Pappalardo et al. 2019; Simpson et al. 2022; Mendes-Neves et al. 2024 | records; Spearman via secondary descriptions only | as cited; none is scored against a betting market |
| Whelan 2025 | record here; the full text was read by the earlier research agent (`reports/open_close/markets_sources.md`) | maker and taker figures |
| Hegarty and Whelan 2025; Whelan and Hegarty 2024 | records and the author's pages | no FLB in AH; refund structures priced differently |
| Constantinou 2022 | abstract fetched (arXiv) | AH shares 1X2's inefficiencies; 13 Premier League seasons |
| Angelini and De Angelis 2019; Angelini, De Angelis and Singleton 2022 | records | eight of eleven markets efficient at best odds; in-play mispricing, reverse FLB, larger after surprises |
| Winkelmann et al. 2021; Deutscher et al. 2018; Bernardo et al. 2019 | records | regime-change mispricings; Bernardo et al. is in QREF, not Finance Research Letters as one secondary source said |
| Grant et al. 2018 | records (Nottingham, Southampton) | position-takers and book-balancers; 545 arbitrage portfolios; SBOBet 77% against Ladbrokes 12.4% |
| Kaunitz et al. 2017 | record and press | consensus strategy; accounts restricted |
| Flepp et al. 2017; Franck et al. 2010; Forrest et al. 2005; Levitt 2004; Strumbelj 2014 | records; Strumbelj through secondary sources | as cited |
| Croxson and Reade 2014; Dixon and Robinson 1998; Robberechts et al. 2021 | records | as cited |
| Benter 1994 (2008 chapter) | record; the PDF found is a scanned image, not readable here | the two-stage method; the figures quoted in secondary accounts disagree and are not cited |
| Thorp 2006 | record | the chapter's existence and subject |

## Searched for and not found

- A peer-reviewed study of Betfair or bookmaker prices around line-up announcements in football.
- A peer-reviewed paper scoring a football model against Pinnacle's or Betfair's close with a positive result.
- A paper on referee assignments and the cards or goals markets.
- Any public description of Jamestown Analytics betting, or of its models.
- Primary descriptions of how Asian brokers and books (Singbet, Crown, SBOBet, IBC) set opening lines.

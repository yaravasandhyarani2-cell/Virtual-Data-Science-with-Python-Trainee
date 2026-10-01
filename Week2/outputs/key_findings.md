# Week 2 - Key EDA Findings: Bike Sharing Demand

## Dataset
- **Source:** UCI / OpenML Bike Sharing Demand (hourly, 2011-2012)
- **Records:** 17,379  |  **Features:** 22

## Top Findings

1. **Peak demand hour: 17:00** (avg 461.5 rides)
   - Demand spikes during commute hours (8 AM and 5-6 PM).

2. **Busiest season: Fall** (avg 236.0 rides/hr)
   - Spring is the slowest season.

3. **Busiest month: Month 9** (avg 240.8 rides/hr)

4. **Rain reduces demand sharply** - Light Rain/Snow avg: 111.6 (45.5% drop vs Clear)

5. **Temperature is the strongest positive correlate** (r = 0.4048)

6. **Humidity negatively correlates** with count (r = -0.3229)

7. **Top 3 peak hours:** ['17', '18', '8']:00

8. **Year-over-year growth: 63.2%** (2011 to 2012)

9. **Day type averages:** {'Holiday': 156.9, 'Weekend': 183.9, 'Working Day': 193.2}

10. **Windspeed IQR outliers:** 342 rows

## Limitations & Next Steps
- Only 2 years of data; no geographic breakdown.
- Next: Feature engineering and predictive modelling (Week 3).

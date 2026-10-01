# Week 2 - Key EDA Findings: Bike Sharing Demand

## Dataset
- **Source:** UCI / OpenML Bike Sharing Demand (hourly, 2011-2012)
- **Records:** 17379  |  **Features:** 13

## Top Findings

1. **Peak demand hour is 17:00**
   - Average of 461.5 rides. Demand spikes during morning (8 AM) and evening (5-6 PM) commutes.

2. **Busiest season: Fall** (avg 236.0 rides/hr)
   - Spring is the slowest season.

3. **Busiest month: Month 9** (avg 240.8 rides/hr)
   - Mid-year summer/fall months drive the highest demand.

4. **Rain sharply reduces demand** - Light Rain/Snow avg: 111.6 (45.5% drop vs Clear)

5. **Temperature is the strongest positive correlate** with count (r = 0.4048)
   - Warmer weather strongly encourages cycling.

6. **Humidity negatively correlates** with count (r = -0.3229)
   - High humidity discourages bike usage.

7. **Top 3 peak hours:** ['17', '18', '8']:00
   - Clear bimodal (commute) pattern on working days; unimodal midday peak on weekends.

8. **Year-over-year growth: 63.2%** (2011 -> 2012)
   - Rapid adoption of the bike-sharing system in its second year.

9. **Average count by day type:** {'Holiday': 156.9, 'Weekend': 183.9, 'Working Day': 193.2}
   - Working days show concentrated rush-hour peaks; weekends show spread leisure usage.

10. **Windspeed outliers (IQR):** 342 records
    - Extreme windspeed values likely correspond to stormy/unusual weather events.

## Limitations & Next Steps
- Dataset covers only 2011-2012; long-term trends cannot be assessed.
- Casual vs registered users show distinct patterns - separate modelling could improve insights.
- Weather granularity is coarse (4 categories); finer meteorological data could improve demand modelling.
- Next: Feature engineering and predictive modelling (Week 3).

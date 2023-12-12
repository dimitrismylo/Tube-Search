
# London Tube Search Algorithms

This project explores a variety of search algorithms applied to the problem of optimal route finding on the London Underground. Namely, breadth-first, depth-first, and uniform cost searches are implemented, with the addition of a variant that adds penalties to the cost function (such as time taken to change tube lines). Additionally a heuristic based search is also implemented, which adds a heuristic for leading the search towards stations that are closer to the zone of the goal station.

The PDF report details how the algorithms were implemented and analyses and compares the performance of them using some example target routes.

## Data File

The file `tubedata.csv` defines the London Tube map in terms of a logical relation, Tube "step". 


Each row in the CSV file represents a Tube “step” and is in the following format:
**[StartingStation], [EndingStation], [TubeLine], [AverageTimeTaken], [MainZone], [SecondaryZone]**
where:
* *StartingStation*: a starting station
* *EndingStation*: a directly connected ending station
* *TubeLine*: the tube line connecting the named stations
* *AverageTimeTaken*: the average time, in minutes, taken between the named stations
* *MainZone*: the main zone of the stations
* *SecondaryZone*: the secondary zone of the stations, which is "0" if the station is in only one zone



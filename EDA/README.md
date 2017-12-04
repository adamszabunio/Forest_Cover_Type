# EDA (Exploratory Data Analysis) and Logistic Regression

![](images/eda.img?raw=true)

EDA and simple model fitting for tree types found in the Roosevelt National Forest in Colorado.
[Forest Cover Type Kaggle Dataset](https://www.kaggle.com/uciml/forest-cover-type-dataset)

[Presentation for First Semester of M.S. Data Science Program](https://github.com/adamszabunio/Forest_Cover_Type/tree/master/EDA/EDA_for_presentation.ipynb)

[Additional Presentation Material](https://github.com/adamszabunio/Forest_Cover_Type/tree/master/EDA/Further_EDA_and_Logistic_Regression.ipynb)

Presentation concludes with a call for Feature Scaling and/or new model selection. 

[Continued Analysis via Decision Tree Based models](https://github.com/adamszabunio/Forest_Cover_Type/tree/master/Random_Forests)

Context
-------
This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections.

Content
-------
There are seven tree types, each represented by an integer variable:

1. Spruce/Fir 
2. Lodgepole Pine 
3. Ponderosa Pine 
4. Cottonwood/Willow 
5. Aspen 
6. Douglas-fir 
7. Krummholz

Remaining data fields include:
-----------------------------
- Elevation: Elevation in meters 
- Aspect: Aspect in degrees azimuth 
- Slope: Slope in degrees 
- Horizontal Distance To Hydrology: Horz Dist to nearest surface water features Vertical Distance To Hydrology: Vert Dist to nearest surface water features 
- Horizontal Distance To Roadways: Horz Dist to nearest roadway 
- Hillshade 9am (0 to 255 index): Hillshade index at 9am, summer solstice 
- Hillshade Noon (0 to 255 index): Hillshade index at noon, summer solstice 
- Hillshade 3pm (0 to 255 index): Hillshade index at 3pm, summer solstice 
- Horizontal Distance To Fire Points: Horz Dist to nearest wildfire ignition points 
- Wilderness Area (4 binary columns, 0 = absence or 1 = presence): Wilderness area designation 
- Soil Type (40 binary columns, 0 = absence or 1 = presence): Soil Type designation 
- Cover Type (7 types, integers 1 to 7): Forest Cover Type designation

The wilderness areas are:
------------------------
1. Rawah Wilderness Area 
2. Neota Wilderness Area 
3. Comanche Peak Wilderness Area 
4. Cache la Poudre Wilderness Area

The soil types are:
------------------

1. Cathedral family - Rock outcrop complex, extremely stony 
2. Vanet - Ratake families complex, very stony 
3. Haploborolis - Rock outcrop complex, rubbly 
4. Ratake family - Rock outcrop complex, rubbly 
5. Vanet family - Rock outcrop complex complex, rubbly 
6. Vanet - Wetmore families - Rock outcrop complex, stony 
7. Gothic family 
8. Supervisor - Limber families complex 
9. Troutville family, very stony 
10. Bullwark - Catamount families - Rock outcrop complex, rubbly 
11. Bullwark - Catamount families - Rock land complex, rubbly. 
12. Legault family - Rock land complex, stony 
13. Catamount family - Rock land - Bullwark family complex, rubbly 
14. Pachic Argiborolis - Aquolis complex 
15. unspecified in the USFS Soil and ELU Survey 
16. Cryaquolis - Cryoborolis complex 
17. Gateview family - Cryaquolis complex 
18. Rogert family, very stony 
19. Typic Cryaquolis - Borohemists complex 
20. Typic Cryaquepts - Typic Cryaquolls complex 
21. Typic Cryaquolls - Leighcan family, till substratum complex 
22. Leighcan family, till substratum, extremely bouldery 
23. Leighcan family, till substratum - Typic Cryaquolls complex 
24. Leighcan family, extremely stony 
25. Leighcan family, warm, extremely stony 
26. Granile - Catamount families complex, very stony 
27. Leighcan family, warm - Rock outcrop complex, extremely stony 
28. Leighcan family - Rock outcrop complex, extremely stony 
29. Como - Legault families complex, extremely stony 
30. Como family - Rock land - Legault family complex, extremely stony 
31. Leighcan - Catamount families complex, extremely stony 
32. Catamount family - Rock outcrop - Leighcan family complex, extremely stony 
33. Leighcan - Catamount families - Rock outcrop complex, extremely stony 
34. Cryorthents - Rock land complex, extremely stony 
35. Cryumbrepts - Rock outcrop - Cryaquepts complex 
36. Bross family - Rock land - Cryumbrepts complex, extremely stony 
37. Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony 
38. Leighcan - Moran families - Cryaquolls complex, extremely stony 
39. Moran family - Cryorthents - Leighcan family complex, extremely stony 
40. Moran family - Cryorthents - Rock land complex, extremely stony

Acknowledgement
--------------
This dataset is part of the UCI Machine Learning Repository, and the original source can be found here. The original database owners are Jock A. Blackard, Dr. Denis J. Dean, and Dr. Charles W. Anderson of the Remote Sensing and GIS Program at Colorado State University.


Citation and sources:
--------------------

[National Park Service - Rocky Mountain National Park Colorado](https://www.nps.gov/romo/learn/nature/conifers.htm)

[Krummholz: The High Life of Crooked Wood](http://northernwoodlands.org/outside_story/article/krummholz-wood)

[An Expression for the Effect of Aspect, Slope, and Habitat Type on Tree Growth](https://www.fs.fed.us/rm/pubs_journals/1976/rmrs_1976_stage_a001.pdf)

[Interactions of Elevation, Aspect, and Slope in Models of Forest Species Composition and Productivity](https://www.fs.fed.us/rm/pubs_other/rmrs_2007_stage_a002.pdf)

[Excellent Map for Visualizing a Selcted Soil Series](https://casoilresource.lawr.ucdavis.edu/see/)

[Soil Taxonomy of Moran Family](https://casoilresource.lawr.ucdavis.edu/soil_web/ssurgo.php?action=explain_component&mukey=766492&cokey=10720693)

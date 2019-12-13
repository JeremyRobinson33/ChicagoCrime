#!/usr/bin/python3
# Jeremy Robinson
# December 2019
# Data Mining Chicago Crime Data
# Required: pip install pandas, maplotlib, numpy, sklearn
# Explore the crimes in Chicago


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import statistics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def plotCrime(crimeType, df, year="default"):
    # make sure the crime type will match
    crimeType = crimeType.upper()
    # get the crime type
    cType = df['Primary Type'] == crimeType

    if year == "default":
        # filter for 2014, 2015, 2016, 2017, and type
        y14 = df[(df.Year == 2014) & cType]

        y15 = df[(df.Year == 2015) & cType]

        y16 = df[(df.Year == 2016) & cType]

        y17 = df[(df.Year == 2017) & cType]

        years = [y14, y15, y16, y17]

        # for each year loop through and make scatter plot of the desired
        # crime
        # Use k means to classify locations
        for y in years:
            X = pd.DataFrame(y.Location.str.replace('(', '').str.replace(')', '').str.split(
                ',', expand=True).values.tolist())

            # run k means on the location
            kmeans = KMeans(n_clusters=4, random_state=0,
                            max_iter=3000).fit_predict(X)

            # calculate the optimal clusters
            distortions = []
            for i in range(1, 11):
                km = KMeans(
                    n_clusters=i, init='random',
                    n_init=10, max_iter=300,
                    tol=1e-04, random_state=0
                )
                km.fit(X)
                distortions.append(km.inertia_)

            # plot
            plt.plot(range(1, 11), distortions, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Distortion')
            plt.show()

            # plot the scatter plot
            colors = ['red', 'green', 'blue', 'purple']

            plt.scatter(y['Longitude'], y['Latitude'], marker='.',
                        c=kmeans, cmap=matplotlib.colors.ListedColormap(colors))

            plt.ylim(41.6, 42.1)
            plt.xlim(-88, -87.4)
            plt.legend(kmeans)
            plt.title(crimeType)
            plt.ylabel('Latitude')
            plt.xlabel('Longitude')
            plt.show()

    # if year is given examine that year and crime type
    else:
        y = df[(df.Year == int(year)) & cType]

        # make the location column usable
        X = pd.DataFrame(y.Location.str.replace('(', '').str.replace(')', '').str.split(
            ',', expand=True).values.tolist())

        # run kmeans
        kmeans = KMeans(n_clusters=4, random_state=0,
                        max_iter=3000).fit_predict(X)

        # elbow method
        distortions = []
        for i in range(1, 11):
            km = KMeans(
                n_clusters=i, init='random',
                n_init=10, max_iter=300,
                tol=1e-04, random_state=0
            )
            km.fit(X)
            distortions.append(km.inertia_)

        # plot
        plt.plot(range(1, 11), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.show()

        colors = ['red', 'green', 'blue', 'purple']

        plt.scatter(y['Longitude'], y['Latitude'], marker='.',
                    c=kmeans, cmap=matplotlib.colors.ListedColormap(colors))

        plt.ylim(41.6, 42.1)
        plt.xlim(-88, -87.4)
        plt.title(crimeType + " " + str(year))
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.show()

# plot histogram for given crime type
# for years 2014, 2015, 2016, and 2017


def plotHist(crimeType, df):
    crimeType = crimeType.upper()

    cType = df['Primary Type'] == crimeType

    year2014 = df[(df.Year == 2014) & cType]
    year2015 = df[(df.Year == 2015) & cType]
    year2016 = df[(df.Year == 2016) & cType]
    year2017 = df[(df.Year == 2017) & cType]

    years = [year2014, year2015, year2016, year2017]

    count = []

    for y in years:
        count.append(len(y.Year))

    mean = statistics.mean(count)

    print("Mean: ", mean)

    print("Standard Deviation: ", (statistics.pstdev(count)))

    print("Variance:  ", (statistics.pvariance(count)))

    c = ['red', 'blue', 'green', 'yellow']

    legend = ['2014', '2015', '2016', '2017']

    # Plot histogram
    plt.hist([years[0]['Primary Type'], years[1]['Primary Type'],
              years[2]['Primary Type'], years[3]['Primary Type']], color=c)
    plt.xlabel("Crime Type")
    plt.ylabel("Frequency")
    plt.legend(legend)
    plt.show()


Crime_Data = pd.read_csv('ChicagoCrimes.csv', na_values=[
                         None, 'NaN', 'Nothing'], header=0)

# Crime_Data = pd.concat(Crime_Data,axis = 0)
Crime_Data.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)

Crime_Data.drop(['Case Number', 'IUCR', 'FBI Code', 'Updated On',
                 'X Coordinate', 'Y Coordinate'], inplace=True, axis=1)

Crime_Data.Date = pd.to_datetime(
    Crime_Data.Date, format='%m/%d/%Y %I:%M:%S %p')
Crime_Data.index = pd.DatetimeIndex(Crime_Data.Date)

# Crime_Data['Primary Type'] = pd.Categorical(Crime_Data['Primary Type'])
# Crime_Data['Description'] = pd.Categorical(Crime_Data['Description'])
# Crime_Data['Location Description'] = pd.Categorical(
#     Crime_Data['Location Description'])

Crime_Data_date = Crime_Data.pivot_table(
    'ID', aggfunc=np.size, columns='Primary Type', index=Crime_Data.index.date, fill_value=0)
Crime_Data_date.index = pd.DatetimeIndex(Crime_Data_date.index)

# remove NAs from Longitude and Latitude data
Crime_Data = Crime_Data.dropna(axis=0, how='any')


crimeType = input("Enter the type of crime you wish to examine ")
year = input("Enter a year to filter by or leave empty ")
plotHist(crimeType, Crime_Data)

if(year == ""):
    plotCrime(crimeType, Crime_Data)
else:
    plotCrime(crimeType, Crime_Data, year)

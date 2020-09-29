import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GoogleNews import GoogleNews
from textblob import TextBlob

already_have_data = True
already_have_sentiments = False
candidates = ['Joe_Biden', 'Bernie_Sanders', 'Elizabeth_Warren', 'Pete_Buttigieg']
time_frames = {"years": [2019, 2020], "month_ranges": [[4, 12], [1, 3]]}
dates = []
# stores average sentiment scores from each month
candidate_monthly_sentiments = {"Joe_Biden": [[], [], 0],
                                "Bernie_Sanders": [[], [], 0],
                                "Elizabeth_Warren": [[], [], 0],
                                "Pete_Buttigieg": [[], [], 0]}
# stores each sentiment score from month
candidate_sentiments_month = {"Joe_Biden": [[], []],
                              "Bernie_Sanders": [[], []],
                              "Elizabeth_Warren": [[], []],
                              "Pete_Buttigieg": [[], []]}

##
# Finds all google news headlines of a given candidate in a certain time frame
# @param candidate expects a String formatted as 'query1+query2+...'
# @param year int for designated year
# @param start_month starting month in month range (1 being Januarary and 12 being December
# @param end_month end month in month range
def find_candidate_headlines(candidate, year, start_month, end_month):
    news = GoogleNews(period='d')

    for i in range(end_month-start_month + 1):
        headlines = {'title': [], 'media': []}
        current_month = start_month + i
        next_month = current_month + 1 if current_month != 12 else 1
        next_year = year if current_month != 12 else year + 1
        headlines_count = 0

        # initializes query w/ keyword and time range
        time_template = '+after:%(year)s-%(current_month)s-01+before:%(next_year)s-%(next_month)s-01'
        time_frame = time_template % {'year': year,
                                      'next_year': next_year,
                                      'current_month': "{:02d}".format(current_month),
                                      'next_month': "{:02d}".format(next_month)}
        query = candidate.replace("_", "+") + time_frame

        start_date = "{:02d}".format(current_month) + '/01/' + str(year)
        end_date = "{:02d}".format(next_month) + '/01/' + str(next_year)
        print(start_date + '    ' + end_date)
        news.setTimeRange(start=start_date, end=end_date)

        news.search(query)
        while headlines_count < 100:
            results = news.result()
            for item in results:
                if item['title']:
                    headlines['title'].append(item['title'])
                    headlines['media'].append(item['media'])

            headlines_count += len(headlines['title'])
            news.clear()

        df = pd.DataFrame(headlines)
        file_path = candidate + "/candidate_headlines/" + candidate + "_" + str(year) + "_" + str(current_month) + '.csv'
        df.to_csv(file_path, mode='a', index=False)


##
# calculates subjectivity and polarity using TextBlob
def calculate_sentiment(candidate, year, start_month, end_month):
    index = candidate_monthly_sentiments[candidate][2]

    for i in range(end_month - start_month + 1):
        current_month = start_month + i
        file_path = candidate + "/candidate_headlines/" + candidate + "_" + str(year) + "_" + str(current_month) + '.csv'
        headlines = pd.read_csv(file_path)
        candidate_monthly_sentiments[candidate][0].append(0)
        candidate_monthly_sentiments[candidate][1].append(0)

        for df_index, row in headlines.iterrows():
            candidate_sentiments_month[candidate][0].append(TextBlob(row["title"]).sentiment[0])
            candidate_sentiments_month[candidate][0][df_index] += TextBlob(row["media"]).sentiment[0]

            candidate_sentiments_month[candidate][1].append(TextBlob(row["title"]).sentiment[1])
            candidate_sentiments_month[candidate][1][df_index] += TextBlob(row["media"]).sentiment[1]

        for article_index in range(len(candidate_sentiments_month[candidate][0])):
            candidate_monthly_sentiments[candidate][0][index] += candidate_sentiments_month[candidate][0][article_index]
            candidate_monthly_sentiments[candidate][1][index] += candidate_sentiments_month[candidate][1][article_index]

        candidate_monthly_sentiments[candidate][0][index] /= headlines.shape[0]
        candidate_monthly_sentiments[candidate][1][index] /= headlines.shape[0]

        index += 1
        graph_sentiment_scores_monthly_histogram(candidate, current_month, year)
        candidate_sentiments_month[candidate] = [[], []]

    candidate_monthly_sentiments[candidate][2] = index


##
# graphs sentiment scores as a scatter plot of cumulative data from each month
def graph_sentiment_scores_scatter():
    cumulative_sentiment = np.zeros((len(candidates), 2))
    for candidate_index in range(len(candidates)):
        candidate = candidates[candidate_index]
        for sentiment_index in range(len(candidate_monthly_sentiments[candidate][0])):
            cumulative_sentiment[candidate_index, 0] += candidate_monthly_sentiments[candidate][0][sentiment_index]
            cumulative_sentiment[candidate_index, 1] += candidate_monthly_sentiments[candidate][1][sentiment_index]

        cumulative_sentiment[candidate_index] /= len(candidate_monthly_sentiments[candidate][0])

    plt.figure(figsize=(12, 6))
    plt.xlabel('Subjectivity', fontweight='bold')
    plt.ylabel('Polarity', fontweight='bold')
    plt.title("Cumulative Sentiment")
    plt.scatter(cumulative_sentiment[:, 0], cumulative_sentiment[:, 1])

    # adds labels to each dot
    for index in range(cumulative_sentiment.shape[0]):
        plt.annotate(candidates[index],
                     (cumulative_sentiment[index, 0], cumulative_sentiment[index, 1]),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')

    plt.savefig("cumulative_sentiment.png")
    plt.clf()


##
# gets all the dates
def calculate_dates():
    temp_dates = []

    for index_year in range(len(time_frames["years"])):
        month_range = time_frames["month_ranges"][index_year]
        start_month = month_range[0]
        end_month = month_range[1]
        year = time_frames["years"][index_year]
        for i in range(end_month - start_month + 1):
            current_month = start_month + i
            temp_dates.append(str(year) + "/" + str(current_month))

    return temp_dates


##
# graphs sentiment scores as a double line graph with sentiments from each month
def graph_sentiment_scores_double_line():
    plt.figure(figsize=(12, 6))
    plt.xlabel('Dates', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Score', fontweight='bold')
    plt.title("Subjectivity Over Time")
    for candidate in candidates:
        subjectivity = candidate_monthly_sentiments[candidate][0]

        plt.plot(dates, subjectivity, label=candidate)

    plt.legend()
    plt.savefig("subjectivity_over_time.png")
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.xlabel('Dates', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Score', fontweight='bold')
    plt.title("Polarity Over Time")
    for candidate in candidates:
        polarity = candidate_monthly_sentiments[candidate][1]
        plt.plot(dates, polarity, label=candidate)

    plt.legend()
    plt.savefig("polarity_over_time.png")


def graph_sentiment_scores_monthly_histogram(candidate, current_month, year):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Subjectivity', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(candidate + " Subjectivity Frequency " + str(year) + "/" + str(current_month))
    plt.hist(candidate_sentiments_month[candidate][0], bins=20)

    file_path = candidate + "/sentiment_plots/" + candidate + "_subjectivity_hist_" + str(year) + "_" + str(current_month) + '.png'
    plt.savefig(file_path)
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.xlabel('Polarity', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(candidate + " Polarity Frequency " + str(year) + "/" + str(current_month))
    plt.hist(candidate_sentiments_month[candidate][1], bins=20)

    file_path = candidate + "/sentiment_plots/" + candidate + "_polarity_hist_" + str(year) + "_" + str(current_month) + '.png'
    plt.savefig(file_path)
    plt.clf()


dates = calculate_dates()
for index_year in range(len(time_frames["years"])):
    month_range = time_frames["month_ranges"][index_year]
    start_month = month_range[0]
    end_month = month_range[1]
    year = time_frames["years"][index_year]

    if not already_have_data:
        for candidate in candidates:
            find_candidate_headlines(candidate, year, start_month, end_month)

    if not already_have_sentiments:
        for candidate in candidates:
            calculate_sentiment(candidate, year, start_month, end_month)

graph_sentiment_scores_double_line()
graph_sentiment_scores_scatter()

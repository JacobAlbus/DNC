import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
from GoogleNews import GoogleNews
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
already_have_data = True
already_have_sentiments = False
already_have_histograms = True
already_have_sentiment_plots = False
candidates = ['Joe_Biden', 'Bernie_Sanders', 'Elizabeth_Warren', 'Pete_Buttigieg']
time_frames = {"years": [2019, 2020], "month_ranges": [[4, 12], [1, 3]]}
dates = []
# stores average sentiment scores from each month
candidate_monthly_sentiments = {"Joe_Biden": [[], [], 0],
                                "Bernie_Sanders": [[], [], 0],
                                "Elizabeth_Warren": [[], [], 0],
                                "Pete_Buttigieg": [[], [], 0]}
# stores all the individual sentiments scores from each month
candidate_cumulative_sentiments = {"Joe_Biden": [[], []],
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


# takes in a candidate's dataframe and removes articles that don't mention them by name
def remove_irrelevant_articles(candidate, df):
    temp = df
    for index, row in temp.iterrows():
        first_name = candidate.split("_")[0]
        last_name = candidate.split("_")[1]
        if (first_name not in row["title"] and last_name not in row["title"]) and (first_name not in row["media"] and last_name not in row["media"]):
            temp = temp.drop(index)

    temp = temp.reset_index()
    return temp


# calculates subjectivity and polarity using TextBlob
def calculate_sentiment(candidate, year, start_month, end_month, already_have_histograms):
    month_index = candidate_monthly_sentiments[candidate][2]

    for i in range(end_month - start_month + 1):
        current_month = start_month + i
        file_path = candidate + "/candidate_headlines/" + candidate + "_" + str(year) + "_" + str(current_month) + '.csv'
        headlines = pd.read_csv(file_path)
        headlines = remove_irrelevant_articles(candidate, headlines)

        candidate_monthly_sentiments[candidate][0].append(0)
        candidate_monthly_sentiments[candidate][1].append(0)
        candidate_cumulative_sentiments[candidate][0].append([])
        candidate_cumulative_sentiments[candidate][1].append([])

        for df_index, row in headlines.iterrows():
            candidate_cumulative_sentiments[candidate][0][month_index].append(analyser.polarity_scores(row["title"])["compound"])
            candidate_cumulative_sentiments[candidate][0][month_index][df_index] += analyser.polarity_scores(row["media"])["compound"]
            candidate_cumulative_sentiments[candidate][0][month_index][df_index] /= 2

            candidate_cumulative_sentiments[candidate][1][month_index].append(TextBlob(row["title"]).sentiment[1])
            candidate_cumulative_sentiments[candidate][1][month_index][df_index] += TextBlob(row["media"]).sentiment[1]
            candidate_cumulative_sentiments[candidate][1][month_index][df_index] /= 2

        candidate_monthly_sentiments[candidate][0][month_index] += stat.mean(candidate_cumulative_sentiments[candidate][0][month_index])
        candidate_monthly_sentiments[candidate][1][month_index] += stat.mean(candidate_cumulative_sentiments[candidate][1][month_index])

        if not already_have_histograms:
            graph_sentiment_scores_monthly_histogram(candidate, current_month, year, month_index)

        month_index += 1

    candidate_monthly_sentiments[candidate][2] = month_index


# finds number of articles for each candidate that don't say them by name in both headline and tagline
def find_num_articles_irrelevant():
    num_articles_non_explicit = {"Joe_Biden": [[], [], 0],
                                 "Bernie_Sanders": [[], [], 0],
                                 "Elizabeth_Warren": [[], [], 0],
                                 "Pete_Buttigieg": [[], [], 0]}

    for index_year in range(len(time_frames["years"])):
        month_range = time_frames["month_ranges"][index_year]
        start_month = month_range[0]
        end_month = month_range[1]
        year = time_frames["years"][index_year]

        for candidate in candidates:
            total_month_index = num_articles_non_explicit[candidate][2]
            other_candidates = []
            for temp in candidates:
                if temp != candidate:
                    other_candidates.append(temp)

            for i in range(end_month-start_month + 1):
                current_month = start_month + i
                file_path = candidate + "/candidate_headlines/" + candidate + "_" + str(year) + "_" + str(current_month) + ".csv"
                df = pd.read_csv(file_path)
                num_articles_non_explicit[candidate][0].append(0)
                num_articles_non_explicit[candidate][1].append(0)

                for index, row in df.iterrows():
                    first_name = candidate.split("_")[0]
                    last_name = candidate.split("_")[1]

                    if(first_name not in row["title"] and last_name not in row["title"]) \
                            and (first_name not in row["media"] and last_name not in row["media"]):
                        num_articles_non_explicit[candidate][0][total_month_index] += 1

                    for other_candidate in other_candidates:
                        other_first_name = other_candidate.split("_")[0]
                        other_last_name = other_candidate.split("_")[1]
                        if (other_first_name in row["title"] or other_last_name in row["title"]) \
                                or (other_first_name in row["media"] or other_last_name in row["media"]):
                            num_articles_non_explicit[candidate][1][total_month_index] += 1

                total_month_index += 1
            num_articles_non_explicit[candidate][2] = total_month_index

    for candidate in candidates:
        graph_num_articles_per_month(num_articles_non_explicit[candidate][0], candidate, "num_articles_non_explicit")
        print(candidate + "  " + str(sum(num_articles_non_explicit[candidate][1])))
        graph_num_articles_per_month(num_articles_non_explicit[candidate][1],
                                     candidate, "num_article_w_other_candidates")


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
    plt.xlabel('Polarity', fontweight='bold')
    plt.ylabel('Subjectivity', fontweight='bold')
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


# graphs sentiment scores as a double line graph with sentiments from each month
def graph_sentiment_scores_double_line():
    plt.figure(figsize=(12, 6))
    plt.xlabel('Dates', fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Score', fontweight='bold')
    plt.title("Subjectivity Over Time")
    for candidate in candidates:
        subjectivity = candidate_monthly_sentiments[candidate][1]
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
        polarity = candidate_monthly_sentiments[candidate][0]
        plt.plot(dates, polarity, label=candidate)

    plt.legend()
    plt.savefig("polarity_over_time.png")


# graphs histogram of subjectivity and polarity for each candidate for each month
def graph_sentiment_scores_monthly_histogram(candidate, current_month, year, month_index):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Polarity', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(candidate + " Polarity Frequency " + str(year) + "/" + str(current_month))
    plt.hist(candidate_cumulative_sentiments[candidate][0][month_index], bins=20)

    file_path = candidate + "/sentiment_plots/" + candidate + "_polarity_hist_" + str(year) + "_" + str(current_month) + '.png'
    plt.savefig(file_path)
    plt.clf()

    plt.figure(figsize=(12, 6))
    plt.xlabel('Subjectivity', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(candidate + " Subjectivity Frequency " + str(year) + "/" + str(current_month))
    plt.hist(candidate_cumulative_sentiments[candidate][1][month_index], bins=20)

    file_path = candidate + "/sentiment_plots/" + candidate + "_subjectivity_hist_" + str(year) + "_" + str(current_month) + '.png'
    plt.savefig(file_path)
    plt.clf()


# graphs histograms of number of articles from each month that don't explicitly name candidate
def graph_num_articles_per_month(num_articles, candidate, file_name):
    plt.figure(figsize=(12, 6))
    plt.xlabel('Date', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(candidate + " Non-Explicit Articles From Each Month")
    plt.bar(dates, num_articles)

    file_path = candidate + "/" + file_name + ".png"
    plt.savefig(file_path)
    plt.clf()


# graphs a histogram of all the the sentiment scores from each month
def graph_cumulative_sentiments_histogram():
    for candidate in candidates:
        total_polarity = []
        total_subjectivity = []

        for polarity_list in candidate_cumulative_sentiments[candidate][0]:
            for value in polarity_list:
                total_polarity.append(value)

        for subjectivity_list in candidate_cumulative_sentiments[candidate][1]:
            for value in subjectivity_list:
                total_subjectivity.append(value)

        plt.figure(figsize=(12, 6))
        plt.xlabel('Polarity', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title(candidate + " Total Polarity Frequency")
        plt.hist(total_polarity, bins=20)

        file_path = candidate + "/" + candidate + "_total_polarity_hist.png"
        plt.savefig(file_path)
        plt.clf()

        plt.figure(figsize=(12, 6))
        plt.xlabel('Subjectivity', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title(candidate + " Total Subjectivity Frequency")
        plt.hist(total_subjectivity, bins=20)

        file_path = candidate + "/" + candidate + "_total_subjectivity_hist.png"
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
            calculate_sentiment(candidate, year, start_month, end_month, already_have_histograms)

if not already_have_sentiment_plots:
    graph_sentiment_scores_double_line()
    graph_sentiment_scores_scatter()
    graph_cumulative_sentiments_histogram()
    find_num_articles_irrelevant()

for candidate in candidates:
    sentiments = candidate_monthly_sentiments[candidate]
    print(candidate + "   polarity_mean   polarity_stdev   subjectivity_mean   subjectivity_stdev")
    print(str(stat.mean(sentiments[0])) + "  " + str(stat.stdev(sentiments[0])) + "  " +
          str(stat.mean(sentiments[1])) + "  " + str(stat.stdev(sentiments[1])) + "  ")




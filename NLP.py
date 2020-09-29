import requests
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
from GoogleNews import GoogleNews
# import stanza
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

candidates = ['Joe_Biden.csv', 'Bernie_Sanders.csv', 'Elizabeth_Warren.csv', 'Pete_Buttigieg.csv']


def create_voting_dicts():
    voting_dict = {'Rep_Yea': [], 'Rep_Nay': [], 'Rep_Pre': [], 'Rep_Nan': [],
                   'Dem_Yea': [], 'Dem_Nay': [], 'Dem_Pre': [], 'Dem_Nan': []}
    year_tracker = 258
    num_congress = 93
    year = 1974
    dem = 0
    rep = 0

    for i in range(36):
        j = 1
        year_URL = 'https://www.govtrack.us/congress/votes?session=' + str(year_tracker+i)\
                   + '&chamber%5B%5D=1&sort=-created&page=1&faceting=false&allow_redirect=true&do_search=1'
        results = requests.get(year_URL).json()['results']
        temp_dict = {'Rep_Yea': [], 'Rep_Nay': [], 'Rep_Pre': [], 'Rep_Nan': [],
                     'Dem_Yea': [], 'Dem_Nay': [], 'Dem_Pre': [], 'Dem_Nan': []}
        print(year_URL)

        temp_page = requests.get("https://www.govtrack.us//congress/votes/"+str(num_congress)+"-"+str(year)+"/s"+str(j+1))
        temp_soup = bs(temp_page.content, 'html.parser')
        temp_heads = temp_soup.findAll("th", {'style': 'padding-left: 6px; padding-right: 6px;'})
        temp_string = temp_heads[0].text.replace(' ', '').rstrip('\n')[3:]
        if temp_string == 'Democrats':
            print('dem')
            dem = 2
            rep = 3
        else:
            print('rep')
            rep = 2
            dem = 3
        print(len(results))
        for item in range(len(results)):
            voting_URL = "https://www.govtrack.us//congress/votes/"+str(num_congress)+"-"+str(year)+"/s"+str(j)
            print(voting_URL)
            voting_page = requests.get(voting_URL)
            voting_soup = bs(voting_page.content, 'html.parser')

            voting_results = voting_soup.findAll("table", {"class": "stats"})
            voting_row_heads = voting_soup.findAll("th", {'scope': 'row'})
            table = voting_results[0]
            col = table.findAll('td')
            num = int(len(col) / len(voting_row_heads))

            temp_list = []
            temp_num = 0
            for string in voting_row_heads:
                s = string.text.replace(' ', '')
                s = s.rstrip('\n')
                s = s[8:]
                temp_list.append(s)
                temp_num += 1
            print(temp_list)
            if 'Yea' in temp_list:
                voting_dict['Rep_Yea'].append(int(col[rep].text))
                voting_dict['Dem_Yea'].append(int(col[dem].text))

                temp_dict['Rep_Yea'].append(int(col[rep].text))
                temp_dict['Dem_Yea'].append(int(col[dem].text))
            if 'Nay' in temp_list:
                voting_dict['Rep_Nay'].append(int(col[rep + num].text))
                voting_dict['Dem_Nay'].append(int(col[dem + num].text))

                temp_dict['Dem_Nay'].append(int(col[dem + num].text))
                temp_dict['Rep_Nay'].append(int(col[rep + num].text))
            if 'NotVoting' in temp_list and len(voting_row_heads) == 3:
                voting_dict['Rep_Nan'].append(int(col[rep + (num * 2)].text))
                voting_dict['Dem_Nan'].append(int(col[dem + (num * 2)].text))

                temp_dict['Rep_Nan'].append(int(col[rep + (num * 2)].text))
                temp_dict['Dem_Nan'].append(int(col[dem + (num * 2)].text))
            elif 'NotVoting' in temp_list and len(voting_row_heads) == 2:
                voting_dict['Rep_Nan'].append(int(col[rep + num].text))
                voting_dict['Dem_Nan'].append(int(col[dem + num].text))

                temp_dict['Rep_Nan'].append(int(col[rep + num].text))
                temp_dict['Dem_Nan'].append(int(col[dem + num].text))
            if 'Present' in temp_list:
                voting_dict['Rep_Pre'].append(int(col[rep + (num * 2)].text))
                voting_dict['Dem_Pre'].append(int(col[dem + (num * 2)].text))

                temp_dict['Rep_Pre'].append(int(col[rep + (num * 2)].text))
                temp_dict['Dem_Pre'].append(int(col[dem + (num * 2)].text))
            if 'NotVoting' in temp_list and len(voting_row_heads) == 4:
                voting_dict['Rep_Nan'].append(int(col[rep + (num * 3)].text))
                voting_dict['Dem_Nan'].append(int(col[dem + (num * 3)].text))

                temp_dict['Rep_Nan'].append(int(col[rep + (num * 3)].text))
                temp_dict['Dem_Nan'].append(int(col[dem + (num * 3)].text))

            if 'Yea' not in temp_list:
                voting_dict['Rep_Yea'].append(0)
                voting_dict['Dem_Yea'].append(0)

                temp_dict['Rep_Yea'].append(0)
                temp_dict['Dem_Yea'].append(0)
            if 'NotVoting' not in temp_list:
                voting_dict['Rep_Nan'].append(0)
                voting_dict['Dem_Nan'].append(0)

                temp_dict['Rep_Nan'].append(0)
                temp_dict['Dem_Nan'].append(0)
            if 'Present' not in temp_list:
                voting_dict['Rep_Pre'].append(0)
                voting_dict['Dem_Pre'].append(0)

                temp_dict['Rep_Pre'].append(0)
                temp_dict['Dem_Pre'].append(0)
            if 'Nay' not in temp_list:
                voting_dict['Rep_Nay'].append(0)
                voting_dict['Dem_Nay'].append(0)

                temp_dict['Rep_Nay'].append(0)
                temp_dict['Dem_Nay'].append(0)

            j += 1

        # temp_df = pd.DataFrame.from_dict(temp_dict)
        # temp_df.to_csv('voting/voting_record_' + str(year) + '.csv', index=False)

        year += 1
        if year % 2 == 1:
            num_congress += 1

    voting_df = pd.DataFrame.from_dict(voting_dict)
    voting_df.to_csv('voting/voting_record.csv', index=False)


def compare_voting_records(state, senator, party):
    year_tracker = 264
    num_congress = 96
    year = 1980
    year_avg = []
    j = 509

    for i in range(36):
        k = 0
        num_agree = 0
        maj_counter = 0

        voting_dict = pd.read_csv('voting/voting_record_' + str(year) + ".csv", delimiter=',', low_memory=False)
        voting_record = np.column_stack([voting_dict[party+'_Yea'].tolist(), voting_dict[party+'_Nay'].tolist()])
        year_URL = 'https://www.govtrack.us/congress/votes?session=' + str(year_tracker) \
                   + '&chamber%5B%5D=1&sort=-created&page=1&faceting=false&allow_redirect=true&do_search=1'
        results = requests.get(year_URL).json()['results']
        print(year_URL)

        print(len(results))

        for item in range(len(results)):
            # if j == 668 and year == 1978:
            #     j += 1
            voting_URL = "https://www.govtrack.us//congress/votes/"+str(num_congress) + "-" + str(year) + "/s" + str(j)
            print(voting_URL)
            voting_page = requests.get(voting_URL)
            voting_soup = bs(voting_page.content, 'html.parser')

            sen_vote = ''
            voting_result = voting_soup.findAll("tr", {"voter_group_1": state})
            if voting_result[0].a.text.split(',')[0] == senator:
                sen_vote = voting_result[0].td.text
            else:
                sen_vote = voting_result[1].td.text

            maj_difference = abs(voting_record[k, 0] - voting_record[k, 1])
            maj_vote = np.amax(voting_record[k])
            party_vote = np.where(voting_record[k] == maj_vote)[0]
            if maj_difference / maj_vote > 0.65:
                maj_counter += 1
                if sen_vote == 'Yea' and party_vote == 0:
                    num_agree += 1
                elif sen_vote == 'Nay' and party_vote == 1:
                    num_agree += 1
                elif sen_vote == 'No Vote':
                    maj_counter -= 1

            j += 1
            k += 1

        year_avg.append(num_agree/maj_counter)

        df = pd.DataFrame(year_avg)
        print(df)
        df.to_csv('voting/' + senator + str(year) + '_voting_record.csv', index=False)

        year += 1
        year_tracker += 1
        if year % 2 == 1:
            num_congress += 1
            j = 1


# defunct, use nibba
def get_headlines(candidate, year, start, end):
    user = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    time_template = '+after:%(year)s-%(month1)s-01+before:%(year)s-%(month2)s-01'

    # creates dictionary to keep track of monthly time frame
    dict_month = {}
    for i in range(end-start):
        month1 = "{:02d}".format(start + i)
        month2 = "{:02d}".format(start + i + 1)
        dict_month[month1] = month2

    # gets results for each month
    for key in dict_month:
        start = 0
        counter = 0
        results = []
        while counter < 101:
            time_frame = time_template % {'year': year,
                                          'month1': key,
                                          'month2': dict_month[key]}
            query = candidate + time_frame

            # query is search term and start allows me to access each page
            url = f'https://news.google.com/search?q={query}&start={start}'
            print(url)

            headers = {"user-agent": user}
            resp = requests.get(url)
            soup = bs(resp.content, "html.parser")

            for g in soup.find_all('article'):
                headline = g.find('h3').text
                print(headline)
                results.append(headline)
                counter += 1
            print(counter)

            start += 10

        print(len(results))
        filename = candidate + '.txt'
        with open(filename, 'w') as f:
            f.write("%s\n" % time_frame.replace('+', ' '))
            for item in results:
                f.write("%s\n" % item)
            f.write("\n")


def nibba(candidate, year, start, end):
    # initializes news object
    news = GoogleNews(period='d')
    num = 0
    headlines = {'title': [],
                 'media': []}
    for i in range(end-start):
        # counter keeps track of number of articles in headlines
        counter = 0

        # initializes query w/ keyword and time range
        time_template = '+after:%(year)s-%(month1)s-01+before:%(year)s-%(month2)s-01'
        time_frame = time_template % {'year': year,
                                      'month1': "{:02d}".format(start + i),
                                      'month2': "{:02d}".format(start + i + 1)}
        query = candidate + time_frame

        # nibba shit
        start_date = "{:02d}".format(start + i) + '/01/' + str(year)
        end_date = "{:02d}".format(start + i + 1) + '/01/' + str(year)
        print(start_date + '    ' + end_date)
        news.setTimeRange(start=start_date, end=end_date)

        # gets ~100 headlines from each month
        news.search(query)
        while counter < 100:
            results = news.result()
            for item in results:
                if item['title']:
                    headlines['title'].append(item['title'])
                    headlines['media'].append(item['media'])
                if item['media'] == '':
                    num += 1

            counter += len(headlines['title'])
            print(counter)
            news.clear()


    df = pd.DataFrame(headlines)
    df.to_csv(candidate + '2.txt', mode='a', index=False)
    print(num)

# num = 0
# for candidate in candidates:
#     index = candidate.index('+')
#     first = candidate[0: index]
#     last = candidate[index + 1: len(candidate)]
#
#     df = pd.read_csv(candidate + '.txt')
#     print(df.shape)
#
#     for i in range(1100):
#         title = df['title'].iloc[i]
#         media = df['media'].iloc[i]
#
#         if first not in title and last not in title:
#             if first not in media and last not in media:
#                 # url = "https://news.google.com/search?q={query}".format(query=title)
#                 # webbrowser.open_new_tab(url)
#                 print(title)
#                 num += 1
#     print(num)


# df = pd.read_csv('Joe+Biden.txt')
# sentences = []
#
# for i in range(10):
#     print(i)
#     sentence_list = []
#
#     sentence = df['title'].iloc[i]
#     for word in sentence.split():
#         sentence_list.append(word)
#
#     sentence = df['media'].iloc[i]
#     for word in sentence.split():
#         sentence_list.append(word)
#
#     sentences.append(sentence_list)
#
# # train model
# model = Word2Vec(sentences, min_count=1)
# # fit a 2d PCA model to the vectors
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
#
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#
# pyplot.show()
# def double_prop(text_file):
#     df = pd.read_csv(text_file)
#     documents = []
#     sentiments = []
#     punctations = [',', ':', ';']
#     negate = ['not', 'but']
#
#     # iterates through df and breaks each headline down into each punctuation zone
#     for i in range(df.shape[0]):
#         zones = [df['title'].iloc[i], df['media'].iloc[i]]
#
#         for p in punctations:
#             for zone in zones:
#                 temp_split = zone.split(p)
#                 if temp_split[0] != zone:
#                     index = zones.index(zone)
#                     zones.pop(index)
#                     for temp_str in reversed(temp_split):
#                         zones.insert(index, temp_str)
#
#         documents.append(zones)
#
#     # creates sentiment list with same dimensions as documents
#     for list in documents:
#         temp = [x if len(list) == -1 else 0 for x in list]
#         sentiments.append(temp)
#
#     # calculates sentiment for each zone for pos/neg
#     sentiments = calc_sentiment('opinion-lexicon/positive-words.csv', 1, documents, negate, sentiments)
#     sentiments = calc_sentiment('opinion-lexicon/negative-words.csv', -1, documents, negate, sentiments)
#
#     # get sentiment for each doc, sort, then get list of neg and pos docs
#     doc_sentiments = []
#     for doc in sentiments:
#         doc_sentiments.append(sum(doc))
#     print(len(doc_sentiments))
#     temp_dict = {'doc_sentiments': doc_sentiments,
#                  'documents': documents,
#                  'zone_sentiments': sentiments}
#     df_sentiments = pd.DataFrame(temp_dict)
#     print(df_sentiments.shape)
#     print(df_sentiments.columns)
#
#     name = candidate.split('.csv')[0]
#     df_sentiments.to_csv(name + '_sentiments.csv')
#     num_neg = 0
#     num_pos = 0
#     for i in range(len(doc_sentiments)):
#         if doc_sentiments[i] < 0:
#             num_neg += 1
#         if doc_sentiments[i] > 0:
#             num_pos += 1
#     nibba = min(num_neg, num_pos)
#

def calc_sentiment(lexicon, sent_word, documents, negate, sentiments):
    # reads in list of sentiment words
    lex = []
    f = open(lexicon, 'r')
    for item in f:
        lex.append(item.split('\n')[0])

    # iterates through list of words, docs, and zones
    for word in lex:
        for doc in documents:
            for zone in doc:
                if word in zone:
                    doc_index = documents.index(doc)
                    zone_index = doc.index(zone)
                    word_index = zone.index(word)
                    is_negate = 1

                    # checks to see if negation word exists
                    if zone[word_index - 1] in negate or zone[word_index - 2] in negate:
                        is_negate = -1
                    # calculates sentiment
                    sentiments[doc_index][zone_index] -= ((len(word) ** 2) / len(zone)) * is_negate * sent_word

    return sentiments


# stanza.download('en')
nltk.download('vader_lexicon')
# nlp = stanza.Pipeline('en')
sia = SIA()

neg_lex = []
f = open('opinion-lexicon/negative-words.csv', 'r')
for item in f:
    neg_lex.append(item.split('\n')[0])

pos_lex = []
f = open('opinion-lexicon/positive-words.csv', 'r')
for item in f:
    pos_lex.append(item.split('\n')[0])

headlines = pd.read_csv('Bernie_Sanders.csv')
titles = headlines['title'].iloc[:10].tolist()
df = pd.DataFrame(columns=['entity', 'sentiment'])

# for word in neg_lex:
#     for sent in sentences:
#         if word in sent:
#             doc = nlp(sent)
#             print(doc.sentences[0].words)
#             index = doc.sentences[0].words.index(word)
#             head = doc.sent.words[index].head
#             print(word + ' modifies ' + head)

for title in titles:
    # doc = nlp(title)
    # print(doc.entities)
    print(sia.polarity_scores(title))
    print()
    # for sent in doc.sentences:
    #     for item in sent.words:
    #         print()

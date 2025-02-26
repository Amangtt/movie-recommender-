import pandas as pd
import matplotlib.pyplot as plt

"""df=pd.read_csv('./Data/content_y_train.csv')
gen=pd.read_csv('./Data/content_bygenre_df.csv')
user=pd.read_csv('./Data/content_user_train.csv')
movies=pd.read_csv('./Data/content_item_train.csv')"""

class analysis:
        def bygenre(df):
            return df
        #bygenre(gen)

        def rating(df):
            plt.hist(df, bins=20)
            plt.title('counts of each rating')
            plt.xlabel("Ratings")
            plt.ylabel("Count")
            plt.show()
        #rating(df)

        def no_of_user(user):
            count=user['user_id'].nunique()
            return count
        #no_of_user(user)

        def no_of_movies(mov):
            count=mov['movie_id'].nunique()
            return count
       

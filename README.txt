Note:

Dataset = please review Marcus

# Beauty
AM1
beautyAmazonData = DataManipulator()  #  5150
beautyAmazonData.load_amazon_review_all_beauty()

AM2
# Book
bookAmazonData = DataManipulator()
bookAmazonData.load_amazon_review_books()

AM3
#Industrial
industrialAmazonData = DataManipulator()
industrialAmazonData.load_amazon_review_industrial_scientific()

AM4
#luxury
luxuryAmazonData = DataManipulator()
luxuryAmazonData.load_amazon_review_luxury_beauty()


AM5
#Magazine
luxuryAmazonData = DataManipulator()
luxuryAmazonData.load_amazon_review_luxury_beauty()

Line 16
TypeAblation='AM1AM4'

Line 41 and 43 to get the best initial result

Line 46 and 51 source and Target

Line 69 and 70 amazonLoaderOld

add ==>  torch.random.seed()  on 72

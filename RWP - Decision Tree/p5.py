from learning import DataSet, DecisionTreeLearner

# Configure the Text's restaurant example, see Figure 18.3
attribute_names = 'Alternate Bar Fri/Sat Hungry Patrons Price Raining Reservation Type WaitEstimate Wait'
restaurant = DataSet(name='restaurant', target='Wait', attrnames=attribute_names)

# Induce a decision tree using the DTL algorithm, see Figure 18.5. 
#random.seed(437) # This random seed gives the text answer 
restaurant_tree = DecisionTreeLearner(restaurant)
restaurant_tree.display()


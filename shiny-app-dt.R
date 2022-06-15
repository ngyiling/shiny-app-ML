############library packages###############
library(caret)
library(shiny)         
library(shinyWidgets) 
library(dplyr)
library(rattle)
library(shinycssloaders)
############data processing################
bank.df<-read.csv("bank-additional-full.csv", stringsAsFactors = TRUE, sep = ";")
sum(is.na(bank.df)) #check missing values
summary(bank.df$y) #check the dataset balance

set.seed(23)
###oversampling
bank_majority <- filter(bank.df, bank.df$y == "no")
bank_minority <- filter(bank.df, bank.df$y == "yes")
bank_majority_sample <- sample_n(bank_majority, 250)
bank_minority_sample <- sample_n(bank_minority, 250)
bank <- rbind(bank_majority_sample, bank_minority_sample)

############set the sample size##############
ChooseSampleSize = function (percentage){
  train.index=createDataPartition(bank$y,p=percentage,list=FALSE)
  return(train.index)
}
##########split train and test dataset############
TrainData = function(train.index){
  i = train.index
  TrainData = bank[i,]
  return(TrainData)
}
TestData = function(train.index){
  i = train.index
  TestData =bank[-i,]
  return(TestData)
}


############input functions################
############decision tree model################
makeDTTree = function(cvnumber,cvrepeat,trainmetric,TrainData) {
  # rpart is performs the split calculations and returns the tree
  traindata = TrainData
  fitcontrol=trainControl(method = "repeatedcv", number = cvnumber,
                          repeats = cvrepeat)
  set.seed(123) 
  DTtree=train(traindata[,-ncol(traindata)],traindata[,ncol(traindata)], 
               method = "rpart", tuneLength=10, metric = trainmetric, 
               trControl = fitcontrol)
  return(DTtree)
}

##############random forest tree model#################
makeRFTree = function(cvnumber,cvrepeat,trainmetric,TrainData){
  traindata = TrainData
  fitControl=trainControl( method = "repeatedcv", number = cvnumber,
                           repeats = cvrepeat)
  set.seed(234) 
  RFtree=train(traindata[,-ncol(traindata)],traindata[,ncol(traindata)],
               method="rf",metric=trainmetric,
               trControl=fitControl,tuneLength=5) # Have a look at the model
  return(RFtree)
}

#############apply test data###################
useDTTree = function(DTtree,TestData) {
  test = TestData
  testdata=test[,-ncol(test)]
  prediction = predict(DTtree, newdata=testdata)
  DTresults = as.data.frame(prediction)
  DTresults$truth = test$y
  
  return(DTresults)
}

useRFTree = function(RFtree,TestData) {
  test = TestData
  testdata=test[,-ncol(test)]
  prediction = predict(RFtree, newdata=testdata)
  RFresults = as.data.frame(prediction)
  RFresults$truth = test$y
  
  return(RFresults)
}

##############calculate the accuracy and TPR/TNR/FPR/FNR/sensivity/specificity#############
calcScores = function(results) {
  results = table(results)
  # calculate the scores on which we'll judge our model to 2 decimal places
  accuracy = round(100 * (results[1] + results[4]) / sum(results), 3)
  true_neg = round(100 * results[1] / sum(results[1, ]), 2)
  true_pos = round(100 * results[4] / sum(results[2, ]), 2)
  false_neg = round(100 * results[3] / sum(results[,2]), 2)
  false_pos = round(100 * results[2] / sum(results[,1]), 2)
  sensitivity = round (results[4]/(results[3]+results[4]),2)
  specificity = round (results[1] / (results[1] + results[2]),2)
  # return the list of result
  return(list(
    paste(c("Overall Accuracy: ",   accuracy, "%"), collapse = ""),
    paste(c("True Positive Rate: ", true_pos, "%"), collapse = ""),
    paste(c("True Negative Rate: ", true_neg, "%"), collapse = ""),
    paste(c("False Positive Rate: ", false_pos, "%"), collapse = ""),
    paste(c("False Negative Rate: ", false_neg, "%"), collapse = ""),
    paste(c("Sensitivity: ", sensitivity), collapse = ""),
    paste(c("Specificity: ", specificity), collapse = "")
  ))
}

################write results as a table#######################
resultsTable = function(results) {
  # shape the result as a table for further output
  
  data = table(results)
  Outcomes = c("Predicted Not Subscribe to the campaign", "Predicted Subscribe to the campaign", "Total")
  # restructure the data and get the actually accept and not accept data
  c1 = c(data[, 1], sum(data[, 1]))  # data[, 1] is a length 2 vector
  c2 = c(data[, 2], sum(data[, 1]))  # data[, 2] is a length 2 vector
  c3 = c(sum(data[, 1]), sum(data[2, ]), sum(data))
  
  # turn these columns back into a dataframe but with proper headers
  output = data.frame(Outcomes)
  output$"Actually Not Subscribe" = c1
  output$"Actually Subscribe"     = c2
  output$"Total"             = c3
  
  return(output)
}

################show CP vs metric plot#######################
DT_plot = function(DTtree) {
  DTplot = plot(DTtree)
  return(DTplot)
}

################show Random forest predictors plot#######################
RF_plot = function(RFtree) {
  RFplot = plot(RFtree)
  return(RFplot)
}

################show random forest variables in plot#######################
RFvarplot = function(RFtree, impvar) {
  varplot = plot(varImp(RFtree), top=impvar)
  return(varplot)
}

# SERVER LOGIC FUNCTION
# Logic of input and output

server = function(input, output, session) {
  # INPUT EVENT REACTIONS
  # reconstruct the decision tree every time when createDTModel is pressed
  # reconstruct the random forest every time when createRFModel is pressed
  
  # get the dt sample size
  DTSampleSize = eventReactive(
    eventExpr = input$createDTModel,
    valueExpr = ChooseSampleSize(input$percentage)
  )
  # get the rf sample size
  RFSampleSize = eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = ChooseSampleSize(input$percentage)
  )
  # get the train and test data for dt and rf
  DTTrainData= eventReactive(
    eventExpr = input$createDTModel,
    valueExpr = TrainData(DTSampleSize()))
  DTTestData= eventReactive(
    eventExpr = input$createDTModel,
    valueExpr = TestData(DTSampleSize()))
  RFTrainData= eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = TrainData(RFSampleSize()))
  RFTestData= eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = TestData(RFSampleSize()))
  
  # build the decision tree model
  DTtree = eventReactive(
    eventExpr = input$createDTModel,
    valueExpr = makeDTTree(input$cvnumber, input$cvrepeat, input$trainmetric,DTTrainData())
  )
  # build the random forest model
  RFtree = eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = makeRFTree(input$cvnumber, input$cvrepeat, input$trainmetric,RFTrainData())
  )
  # generate test results
  DTtest_results = eventReactive(
    eventExpr = input$createDTModel,
    valueExpr = useDTTree(DTtree(),DTTestData())
  )
  RFtest_results = eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = useRFTree(RFtree(),RFTestData())
  )
  
  # generate plots
  
  # random forest variance importance plot
  RFvarimp_plot = eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = RFvarplot(RFtree(),input$impvar)
  )
  # decision tree plot
  DTplot = eventReactive(
    eventExpr = input$createDTModel,
    valueExpr = DT_plot(DTtree())
  )
  # random forest predictors plot
  RFplot = eventReactive(
    eventExpr = input$createRFModel,
    valueExpr = RF_plot(RFtree())
  )
  
  # OUTPUT DISPLAY
  # assessment scores are each collapsed to display on a new line
  output$DTtest_scores = renderText(
    paste(calcScores(DTtest_results()), collapse = "\n")
  )
  output$RFtest_scores = renderText(
    paste(calcScores(RFtest_results()), collapse = "\n")
  )
  
  # tables of outcome breakdows are static widgets
  output$DTtest_table = renderTable(
    resultsTable(DTtest_results()),
    align = "lccc",  # left-align first column, centre rest
    striped = TRUE
  )
  output$RFtest_table = renderTable(
    resultsTable(RFtest_results()),
    align = "lccc",  # left-align first column, centre rest
    striped = TRUE
  )
  
  # frame for a plot of the decision tree
  output$tree_plot = renderPlot(
    fancyRpartPlot(
      DTtree()$finalModel
    )
  )
  # random forest important variable plot
  output$rf_varimp_plot = renderPlot(
    RFvarimp_plot()
  )
  # decision tree complexity parameter and metrics plot
  output$cp_plot = renderPlot(DTplot())
  
  #random forest plot
  output$rf_plot = renderPlot(RFplot())
  
  # best cp value picked
  output$cp_value = renderText(
    paste("The CP value picked for the optimal model: ", DTtree()$bestTune)
  )
  
  # best mtry value picked
  output$mtry_value = renderText(
    paste("The number of randomly selected predictors of the optimal model: ", RFtree()$bestTune)
  )
}

# USER INTERFACE FUNCTION

ui = fluidPage(
  # title
  titlePanel("Bank Marketing Campaign Effectiveness - a Classification Study "),
  helpText("The dataset consists of direct marketing campaigns of a Portuguese bank, 
           and the aim is to predict whether the client will subscribe ('yes') or not ('no') 
           to a term deposit based on phone calls to the client. This study uses two 
           classification approaches - Decision Tree and Random Forest, to explore this dataset."),
  # sidebar ui design
  sidebarLayout(
    sidebarPanel(
      h2("The Controls"),
      br(),
      
      # build the action button
      actionButton(
        inputId = "createDTModel",
        label = "Create Decision Tree Model",
        class = "btn-primary"  # makes it blue!
      ),
      br(),
      br(),
      
      actionButton(
        inputId = "createRFModel",
        label = "Create Random Forest Model",
        class = "btn-primary"  # makes it blue!
      ),
      
      h3("Model Features"),
      helpText(
        "The model features can be adjusted to look at how different variables",
        " affect the prediction outcomes.",
        "These features include Dataset Split, Train Metric Choice of Kappa and Accuracy,",
        "Cross-Validation Number, Cross-Validation Repeat, and Random Forest",
        "Top Important Variable."
      ),
      br(),
      
      # set the slider and picker for input variables
      h4("DatasetSplit"),
      helpText("The proportion of original data used to train the model."),
      sliderInput(
        inputId = "percentage",
        label = NULL,  
        min = 0,       
        max = 1,      
        value = 0.7      
      ),
      br(),
      h4("Decision Tree Train Metric Choice"),
      helpText("Accuracy is the percentage of correctly predicted instances out of all instances in the testing set."),
      helpText("Kappa is a metric to compare an observed accuracy with an expected accuracy at 
               random chance."),
      pickerInput(
        inputId = "trainmetric",
        label = NULL,  
        # choose from accuracy and kappa for the metrics
        choices = list("Accuracy","Kappa"), 
        selected = "Accuracy",
        options = list(`actions-box` = TRUE),
        multiple = FALSE
      ),
      br(),
      h4("Cross-Validation Number"),
      helpText("Cross-Validation is a resampling method to test and 
               train the classfier on different partitions of the trainig data in order to reduce error rate.
               Cross-validation number is the number of partitions that the training data is to be split to."),
      sliderInput(
        inputId = "cvnumber",
        label = NULL,  
        min = 1,       
        max = 30,      
        value = 10      
      ),
      br(),
      h4("Cross-Validation Repeat"),
      helpText("The number of times to repeat the cross-validation process."),
      sliderInput(
        inputId = "cvrepeat",
        label = NULL,  
        min = 1,       
        max = 30,      
        value = 3      
      ),
      br(),
      h4("Random Forest Top Important Variable"),
      helpText("The number of variables to display on the important variables graph."),
      sliderInput(
        inputId = "impvar",
        label = NULL,  
        min = 3,       
        max = 30,      
        value = 15      
      ),
      br(),
      helpText("**click Create Model again after adjusting the controls**"),
      br(),
      h3("Notes:"),
      br(),
      HTML("<b>Overall Accuracy</b> measures the percentage of correct predictions."),
      br(),
      HTML("<b>True Positive Rate</b> is the probability that an actual positive (client actually subscribed) is 
           predicted as positive (client predicted as subscribe)."),
      br(),
      HTML("<b>True Negative Rate</b> is the probability that an actual negative (client actually did not subscribe)
           is predicted as negative (client predicted as not subscribe."),
      br(),
      HTML("<b>False Positive Rate</b> is the probability that an actual negative is predicted as positive."),
      br(),
      HTML("<b>False Negative Rate</b> is the probability that an actual positive is predicted as negative."),
      br(),
      HTML("<b>Sensitivity</b> measures the percentage of true positives correctly predicted."),
      br(),
      HTML("<b>Specificity</b> measures the percentage of true negatives correctly predicted."),
    ),
    mainPanel(
      # set the mainpanel
      fluidRow(
        label = NULL,
        h1("Decision Tree"),
        helpText(
        "This is a supervised machine learning method to segment the data features into smaller, non-overlapping regions
                 in order to classify observations. In the decision process, the tree uses a recursive binary split to assign 
                 each observation to a specific category beginning at the top of the tree. The terminal nodes 
                (nodes at the bottom) shows the final outcomes of the tree that cannot be futher categorized and contains a prediction class label."
        ),
        br(),
        column(6,
               h3("Complexity Parameter(CP) vs Metrics"),
               helpText(
                 "This plot shows the relationship between the complexity parameter (cp) and the variation of the train metric choice (Accuracy or Kappa) 
                 on the selected classification model."
                 ),
               helpText("Cp is defined as the minimum improvement required in each decision tree node, whereby cp will stop 
                 the tree building if the costs of adding another node exceed the cp value. Hence, cp can help determine the optimal tree size and evaluate 
                 the model's performance. A small cp value will result in overfitting, and a large cp value will result in underfitting, giving a poor model
                 performance."
               ),
               tagAppendAttributes(
                 textOutput("cp_value"),
                 style = "white-space: pre-wrap; font-size: 14px;"
               ),
               br(),
               withSpinner(plotOutput(outputId = "cp_plot"),type=4),
               br(),
        ),
        column(6,
               h3("Test Results"),
               helpText(
                 "This table evaluates the performance and the outcomes of the Decision Tree model on the testing data.
                 See notes for description of the measures. "
               ),
               tagAppendAttributes(
                 textOutput("DTtest_scores"),
                 style = "white-space: pre-wrap; font-size: 17px;"
               ),
               br(),
               withSpinner(tableOutput("DTtest_table"),type=4)
        )
      ),
      # plot of the decision tree
      h3("Decision Tree Plot"),
      helpText(
        "This plot visualises the Decision Tree model on the testing data and its decision process."
      ),
      withSpinner(plotOutput(outputId = "tree_plot"),type=4),
      br(),
      
      # plot of random forest
      fluidRow(
        label = NULL,
        h1("Random Forest"),
        helpText("Random Forest is a classification method that consists of many decision trees. This method uses randomly 
        sampled number of features to build each individual tree and each tree gives out a class prediction.
        The class with the most votes will then become the model's prediction. In order to minimise the features correlation and to reduce the variance,  
        the optimal number of features to select for random forest is the sqrt of total feature numbers."),
        br(),
        column(6,
               h3("Random Forest Plot"),
               helpText(
                 "This plot shows the accuracy of the model against the number of randomly selected features with
                 cross validation."
               ),
               tagAppendAttributes(
                 textOutput("mtry_value"),
                 style = "white-space: pre-wrap; font-size: 14px;"
               ),
               br(),
               withSpinner(plotOutput(outputId = "rf_plot"),type=4),
               br(),
        ),
        column(6,
               h3("Random Forest Test Results"),
               helpText(
                 "This table evaluates the performance and the outcomes of the random forest model on the testing data. 
                 See notes for description of the measures."),
               tagAppendAttributes(
                 textOutput("RFtest_scores"),
                 style = "white-space: pre-wrap; font-size: 17px;"
               ),
               br(),
               withSpinner(tableOutput("RFtest_table"),type=4)
        )
      ),
      h3("Important Variables"),
      helpText(
        "This plot shows the variables that has the most predictive power. 
        Variables with high importance have a significant impact on predicting
        the outcomes and they are the key drivers in accurate predictions. Alternatively,
        the model relies less on variables with low importance to make predictions and 
        they may be excluded from the model to enable a faster prediction."
      ),
      withSpinner(plotOutput(outputId = "rf_varimp_plot"),type=4),
      br(),
    )
  )
)



options(shiny.port = 8100)  # when running locally
shinyApp(ui = ui, server = server)


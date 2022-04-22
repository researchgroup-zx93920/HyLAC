Note:  Orignal message sent on Slack from Samiran to Wes.  Language needs to be updated

## LAP:

 - no. of times each step is invoked. use the getstepcounts() function. (There are 6 steps in total. (check the solve function of cuLAP class) [Stacked Bar graph would be fine for this]
 - Time taken by each step (all invocations summed up).. (Something like a pie chart should help). Use the getsteptimes() function.
 -For the step with largest time, check how many kernels it has and get time taken by each kernel. (I can help for this stage if you find CUDA difficult to work with.
 - Histogram on number of parallel pivots done. (This will need a meetup to discuss)
 - All these trends to be plotted based on problem size.

## HungarianLAP:

 - no. of times each step is invoked. (There is no easy function here) but the kernel calls do record this data)
 - Time taken by each step. (same as above)
 - Histogram on number of paralle pivots

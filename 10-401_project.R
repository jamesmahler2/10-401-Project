#setwd("C:\\Users\\Owner\\Desktop\\CMU\\Spring2018\\10-401\\homework\\project")
library(forecast)
library(astsa)

get_prediction = function(arima_order, train, test, is_log)
{
    if(is_log)
    {
        train = log(train)
    }
    model = arima(x=train, order=arima_order, optim.control=list(maxit=1000), method = "ML")
    fc = forecast(model, h=length(test))
    if(is_log)
    {
        fc$mean = exp(fc$mean)
    }
    return(fc$mean)
}

going_forward = function(start, step, stop, model_params, df, is_log)
{
    stopifnot((stop - start) %% step == 0)
    x = (start+1):length(df)
    truth = df[x]
    preds = c()
    folds = c()
    i = 0
    while(start + step * (i + 1) <= stop)
    {
        train_stop = start + i * step
        pred = get_prediction(model_params, train=df[1:(train_stop)],
                              test=df[(train_stop+1):(train_stop+step)],
                              is_log=TRUE)
        i = i + 1
        preds = c(preds, pred)
        folds = c(folds, rep(i, step))
    }
    return(data.frame(x, preds, truth, folds))
}

get_mean_abs_err = function(df)
{
    # assumes df has columns preds, and truth
    return(mean(abs(df$preds - df$truth)))
}

# ARIMA for birth
# plot data
png(filename="birth.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(birth, type="l", main="US Monthly Live Births",
     ylab="Monthly Births (thousands)", xlab="Date")
grid()
dev.off()

# plot transformed data
png(filename="birth_logreturn.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(diff(log(birth)), type="l", main="Log-return US Monthly Live Births")
abline(h=0, col="red")
grid()
dev.off()

# acf - order up to 12
png(filename="acf_birth_logreturn.png", units="in", width=8, height=6, pointsize=12, res=256)
acf(diff(log(birth)), main="ACF of log-return US Monthly Live Births")
dev.off()

# pacf - order up to 12
png(filename="pacf_birth_logreturn.png", units="in", width=8, height=6, pointsize=12, res=256)
pacf(diff(log(birth)), main="PACF of log-return US Monthly Live Births")
dev.off()

birth_model_df = data.frame(c(NA), c(NA), c(NA), c(NA), c(NA))
names(birth_model_df) = c("AR", "I", "MA", "Forecast Error", "AIC")
birth_model_df= na.omit(birth_model_df)
for(p in 1:12)
{
    for(q in 1:12)
    {
        birth_pred_df = going_forward(248, 25, 373, c(p, 1, q), birth, is_log=TRUE)
        model = arima(x=log(birth), order=c(p, 1, q), optim.control=list(maxit=1000), method="ML")
        fc_error = get_mean_abs_err(birth_pred_df)
        birth_model_df[nrow(birth_model_df) + 1,] = list(p, 1, q, fc_error, model$aic)
    }
}
png(filename="birth_error.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(y=birth_model_df$`Forecast Error`, x=(birth_model_df$AR + birth_model_df$MA),
     main="US Monthly Live Births Mean Absolute Forecast Error", xlab="Total Features",
     ylab="Forecast Error")
grid()
dev.off()
# ARIMA(4, 1, 3) is the best as far as predictors vs error
# length(birth) == 373
# step = 25, 4 folds
birth_arima_preds = going_forward(248, 25, 373, c(4, 1, 3), birth, is_log=TRUE)
png(filename="birth_cv.png", units="in", width=8, height=6, pointsize=12, res=256)
#plot(x=1:373, y=birth, type="l", lwd=2, xlim=c(100, 373),
#     main="US Monthly Births - Going Forward ARIMA(4, 1, 3)",
#     xlab="Time", ylab="Births (thousands)"); grid()
plot(birth, type="l", lwd=2, xlim=c(1948 + 100/12, 1979),
     main="US Monthly Births - Going Forward ARIMA(4, 1, 3)",
     xlab="Time", ylab="Births (thousands)"); grid()
colors = rep(c("red", "blue", "orange", "purple"), 5)
for(i in 1:5)
{
    df = birth_arima_preds[birth_arima_preds$folds == i,]
    lines(x=(1948 - 1/12 + df$x/12), y=df$pred, col=colors[i], lwd=2)
    #lines(x=(df$x), y=df$pred, col=colors[i], lwd=2)
}
dev.off()
get_mean_abs_err(birth_arima_preds)

# ARIMA for Johnson Johnson
# plot data
png(filename="jj.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(jj, type="l", ylab="Earnings/Share", xlab="Date",
     main="Johnson and Johnson Quarterly Earnings/Share")
grid()
dev.off()

# plot transformed data
png(filename="jj_logreturn.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(diff(log(jj)), type="l", main="Log-return Johnson and Johnson Quarterly Earnings/Share")
abline(h=0, col="red")
grid()
dev.off()

# acf - order up to 4
png(filename="acf_jj_logreturn.png", units="in", width=8, height=6, pointsize=12, res=256)
acf(diff(log(jj)), main="ACF of log-return J&J Earnings/Share")
dev.off()

# pacf - order up to 4
png(filename="pacf_jj_logreturn.png", units="in", width=8, height=6, pointsize=12, res=256)
pacf(diff(log(jj)), main="PACF of log-return J&J Earnings/Share")
dev.off()

# perform cross validation on models
jj_model_df = data.frame(c(NA), c(NA), c(NA), c(NA), c(NA))
names(jj_model_df) = c("AR", "I", "MA", "Forecast Error", "AIC")
jj_model_df= na.omit(jj_model_df)
for(p in 1:4)
{
    for(q in 1:4)
    {
        jj_pred_df = going_forward(59, 5, 84, c(p, 1, q), jj, is_log=TRUE)
        model = arima(x=log(jj), order=c(p, 1, q), optim.control=list(maxit=1000), method="ML")
        fc_error = get_mean_abs_err(jj_pred_df)
        jj_model_df[nrow(jj_model_df) + 1,] = list(p, 1, q, fc_error, model$aic)
    }
}
# generalization
png(filename="jj_error.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(y=jj_model_df$`Forecast Error`, x=(jj_model_df$AR + jj_model_df$MA),
     main="J&J Mean Absolute Forecast Error", xlab="Total Features",
     ylab="Forecast Error")
grid()
dev.off()

# ARIMA(4, 1, 1) is best as far as predictors vs error
jj_arima_preds = going_forward(59, 5, 84, c(4, 1, 1), jj, is_log=TRUE)
#plot(x=1:84, y=jj, type="l", lwd=2); grid()
png(filename="jj_cv.png", units="in", width=8, height=6, pointsize=12, res=256)
plot(jj, type="l", lwd=2, ylab="Earnings/Share",
     main="J&J Earnings/Share - Going Forward ARIMA(4,1,1)")
grid()
colors = rep(c("red", "blue", "orange", "purple"), 5)
for(i in 1:5)
{
    df = jj_arima_preds[jj_arima_preds$folds == i,]
    lines(x=(1959.75 + 0.25 * df$x), y=df$pred, col=colors[i], lwd=2)
}
dev.off()
get_mean_abs_err(jj_arima_preds)

##########################
# Implementing baseline prediction
##########################

#Use last point to predict
baseline_predict <- function(df, start, step, stop){
  reps <- (stop - start) / step
  i = 0
  preds <- c()
  while(i < reps){
    preds <- c(preds, rep(df[(start+(i*step))], step))
    i <- i+1
  }
  mean_abs_error <- mean(abs(df[(start+1):stop] - preds))
  return(preds)
}

#Use averge of last 5 to predict
baseline_predict_2 <- function(df, start, step, stop){
  reps <- (stop - start) / step
  i = 0
  preds <- c()
  while(i < reps){
    new_preds <- rep(mean(df[(start-5+(i*step)):(start+(i*step))]), step)
    preds <- c(preds, new_preds)
    i <- i+1
  }
  mean_abs_error <- mean(abs(df[(start+1):stop] - preds))
  return(preds)
}

#Make plots with ARIMA predictions and baseline predictions:

birth_base_preds <- baseline_predict(birth, 248, 25, 373) # Mean absolute error is 18.864
birth_base2_preds <- baseline_predict_2(birth, 248, 25, 373) # Mean absolute error is 16.03333
birth_arima_preds <- going_forward(248, 25, 373, c(4, 1, 3), birth, is_log=TRUE)
svg(filename="birth_cv_base.svg", width=8, height=6, pointsize=12)
plot(birth, type="l", lwd=2, xlim=c(1948 + 100/12, 1979),
     main="US Monthly Births - Going Forward ARIMA(4, 1, 3)",
     xlab="Time", ylab="Births (thousands)"); grid()
for(i in 1:5)
{
  df = birth_arima_preds[birth_arima_preds$folds == i,]
  lines(x=(1948 - 1/12 + df$x/12), y=df$pred, col = "purple", lwd=2)
  lines(x = (1948 - 1/12 + df$x/12), y = birth_base_preds[((25*(i-1)+1)):(25*(i))], col = "green", lwd = 2)
  lines(x = (1948 - 1/12 + df$x/12), y = birth_base2_preds[((25*(i-1)+1)):(25*(i))], col = "orange", lwd = 2)
  legend(1970, 400, legend = c("Baseline Method 1", "Baseline Method 2", "ARIMA Predictions"),
         col = c("green", "orange", "purple"), lty = 1, cex = 1.2)
}
dev.off()

jj_base_preds <- baseline_predict(jj, 59, 5, 84) # Mean absolute error is 1.6164
jj_base2_preds <- baseline_predict_2(jj, 59, 5, 84) # Mean absolute error is 2.0442
jj_arima_preds <- going_forward(59, 5, 84, c(4, 1, 1), jj, is_log=TRUE)
svg(filename="jj_cv_base.svg", width=8, height=6, pointsize=12)
plot(jj, type="l", lwd=2, ylab="Earnings/Share",
     main="J&J Earnings/Share - Going Forward ARIMA(4,1,1)")
grid()
for(i in 1:5)
{
  df = jj_arima_preds[jj_arima_preds$folds == i,]
  lines(x=(1959.75 + 0.25 * df$x), y=df$pred, col = "purple", lwd=2)
  lines(x=(1959.75 + 0.25 * df$x), y=jj_base_preds[(5*(i-1)+1):(5*(i))], col = "green", lwd=2)
  lines(x=(1959.75 + 0.25 * df$x), y=jj_base2_preds[(5*(i-1)+1):(5*(i))], col = "orange", lwd=2)
  legend(1961, 15, legend = c("Baseline Method 1", "Baseline Method 2", "ARIMA Predictions"),
         col = c("green", "orange", "purple"), lty = 1, cex = 1.2)
}
dev.off()

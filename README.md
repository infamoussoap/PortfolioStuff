# Constrained Portfolio Optimization
In the constrained portfolio optimization, I only have a fixed amount to invest with, say $1,000. This project will
then find how to best spend that $1,000 over different stocks to minimize the historic volatility, and maximise the 
historic returns.

Note that how well a portfolio performs historically, this is not an indication on how well it will do in the future.

### Precursor
First, this is not financial advice, nor is it investment advice. I did not graduate in finance, nor am I a professional
in the financial sector. I am just a PhD student who likes machine learning, and knows a very little about timeseries.
This project is really just something fun, that I created in ~1 week, that fuses 3 areas I'm good at: 
1. Machine Learning, 2. Math, 3. Programming

### Modern Portfolio Theory
Secondly, I have no idea if this project will ever be useful, especially because we have modern portfolio theory. 
In particular, modern portfolio has what is known as the efficient frontier. This (putting it very briefly) 
is the most optimal portfolio one can have that both minimise the volatility, and maximise the returns of a portfolio.

### Portfolio Optimization
This project is not that. The efficient frontier requires one to be actively buying and selling the shares one owns. I,
on the other hand, have no interest in selling the shares I own. Given my current portfolio, I want it to find how much
of a given stock I should buy to minimise the volatility and maximise returns of the portfolio.

While I have written code for this, I do believe that you can simply use the efficient frontier for this. Some sleight
of hand will be required, though.

### Constrained Portfolio Optimization
It is really the constrained portfolio optimization that I am interested in. 

As an individual investor, I do not have limitless amounts of money to re-balance my portfolio, unlike institutional 
investors. I only have a fixed amount, say $1,000, to invest with. The new problem is then, given I only have $1,000
what is the combination of stocks I can buy to minimise the volatility and maximise the returns.

From my 1 minute of research, I have not seen anyone else do this, but I'm most likely very wrong. I also don't really
see how one can alter the efficient frontier to solve this problem.

### Package Requirements
If you installed python through anaconda, then you should already have the libraries required. You will only need to 
install `yfinance`

The main packages are: `numpy, pandas, yfinance, datetime`

### 
Some understanding of gradient descent would be good, in particular the understanding of epochs, learning rate, and
learning how to see if you have convergence
= Fourier Series
This section assumes that you have already watch #underline(text(blue, link("https://www.youtube.com/watch?v=spUNpyF58BY")[3b1b's video on Fourier transformation])). You should know where this formula comes from and what does it mean:
$
  X (omega) = integral_RR x(t) e^(-i omega t) dif t,
$
where $omega = 2 pi f$ and $f$ is the frequency in Hertz.

Let's start with a *periodic function*, $x(t)$ with a period $T$. This means that $x(t)$ is completely defined by $t in [0, T]$, since $x(t + T) = x(t)$. If we wish to see what frequencies are present in this periodic function, we may perform a Fourier transform on it:
$
  X (omega) = & integral_RR x(t) e^(-i omega t) dif t
$
We can split the integral into intervals of length $T$ and rewrite the integrals so that they all the same bounds:
$
  X (omega) = & dots.c + integral_(-T/2 - T)^(-T/2) x(t) e^(-i omega t) dif t + integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t + integral_(T/2)^(T/2 + T) x(t) e^(-i omega t) dif t + dots.c\
  = & dots.c + integral_(-T/2)^(T/2) x(t - T) e^(-i omega (t - T)) dif t + integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t + integral_(-T/2)^(T/2) x(t + T) e^(-i omega (t + T)) dif t + dots.c\
  = & sum_(k = -oo)^(oo) integral_(-T/2)^(T/2) x(t + k T) e^(-i omega (t + k T)) dif t.
$
We now take the advantage of the periodicity of $x(t)$, and replace $x(t + k T)$ with $x(t)$:
$
  X (omega) = & sum_(k = -oo)^(oo) integral_(-T/2)^(T/2) x(t) e^(-i omega (t + k T)) dif t \
            = & sum_(k = -oo)^(oo) integral_(-T/2)^(T/2) x(t) e^(-i omega t) e^(-i omega k T) dif t \
            = & sum_(k = -oo)^(oo) (e^(-i omega k T) integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t).
$
Since the integral does not depend on $k$, we can take it out of the summation:
$
  X (omega) = & (integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t) (sum_(k = -oo)^(oo) e^(-i omega k T)).
$
We shall not prove it here, but as in the 3b1b's video, we can assume that the summation
$
  sum_(k = -oo)^(oo) e^(-i omega k T)
$
is going to be zero for most values of $omega$, except for $omega_0 = 2 pi 1 / T$ and its multiples. In other words, the summation (_and thus $X(omega)$_) is non-zero only when $omega = n omega_0$ for some integer $n$.\

Now we let's consider $X (n omega_0)$ since all other values of $omega$ are zero:
$
  X (n omega_0) = (integral_(-T/2)^(T/2) x(t) e^(-i n omega_0 t) dif t) (sum_(k = -oo)^(oo) e^(-i n omega_0 k T)).\
$
Here,
$
  sum_(k = -oo)^(oo) e^(-i n omega_0 k T) = sum_(k = -oo)^(oo) e^(-i 2 pi n k) = sum_(k = -oo)^(oo) 1 = oo,
$
the value $X(n omega_0)$ blows up to infinity. Our attempt to find the Fourier transform of a periodic function has led us to a weird function which is *zero everywhere except for $omega = n omega_0$ where it is infinite*. This is not a function in the usual sense, in order to work with it, mathematicians have invented a new kind of functions, from which the most famous one is the Dirac delta function, $delta(x)$, defined as
$
  delta (x) = cases(0 quad x != 0, oo quad x = 0),
$
with the property that
$
  integral_(RR) delta (x) dif x = 1 "and that" integral_(RR) f(x) delta (x - a) dif x = f(a).
$
Intuitively speaking, the Dirac delta function is a function that has a big _spike_ at $x = 0$. And our $X(omega)$ is just a function that has _spikes_ at $omega = n omega_0$ for all integers $n$. Hence, we should be able to find a way to express $X(omega)$ in terms of the Dirac delta function. However, we need to be careful with the _height_ of the spikes even though they are infinite. For example,
$
  integral_(RR) delta (x) dif x = 1 quad integral_(RR) 2 delta (x) dif x = 2,
$
even though both $delta(x)$ and $2 delta(x)$ are infinite at $x = 0$. We are going to omit the proof here, but it can be shown that
$
  sum_(k = -oo)^(oo) e^(-i omega k T) = (2 pi) / T sum_(k = -oo)^(oo) delta (omega - k omega_0).
$
The summation of delta functions on the right-hand side is called the *_Dirac comb_ function* or an *impulse train*.
Therefore,
$
  X (omega) = (integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t) ((2 pi) / T sum_(n = -oo)^(oo) delta (omega - n omega_0)) \
  = (2 pi) / T (integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t) (sum_(k = -oo)^(oo) delta (omega - k omega_0)).
$
Now, we can reconstruct $x(t)$ from $X(omega)$ by taking the inverse Fourier transform:
$
  x(t) = & 1/(2 pi) integral_(RR) X (omega) e^(i omega t) dif omega \
  = & 1/(2 pi) integral_(RR) (2 pi) / T (integral_(-T/2)^(T/2) x(t) e^(-i omega t) dif t) (sum_(k = -oo)^(oo) delta (omega - k omega_0)) e^(i omega t) dif omega \
$
Due to the property of the Dirac delta function, the integral over $RR$ becomes a summation of the integrand evaluated at $omega = k omega_0$:
$
  x(t) = & sum_(k = -oo)^(oo) (1/T integral_(-T/2)^(T/2) x(t) e^(-i k omega_0 t) dif t) e^(i k omega_0 t). \
$
For the sake of convenience, we define
$
  c_k = 1/T integral_(-T/2)^(T/2) x(t) e^(-i k omega_0 t) dif t,
$
so that we can write
$
  x(t) = sum_(k = -oo)^(oo) c_k e^(i k omega_0 t).
$
This is the Fourier series representation of a periodic function $x(t)$ with period $T$. The coefficients $c_k$ are known as the Fourier coefficients and they capture the contribution of each frequency component $k omega_0$ to the overall signal.

= Data Sampling Process
Let's say that the true signal is $x(t)$, and we want to sample it at a rate of $f_s = 1/T_s$ samples per second. A way to mathematically model the sampling process is to multiply $x(t)$ with an *impulse train* $p(t)$, which is a periodic function with period $T_s$, defined as
$
  p(t) = sum_(n = -oo)^(oo) delta (t - n T_s).
$
Since we know that it is periodic with period $T_s$, we can write its Fourier series representation as
$
  p(t) = sum_(k = -oo)^(oo) c_k e^(i k omega_s t),
$
where $omega_s = 2 pi f_s$ is the angular frequency of the sampling process, and $c_k$ are the Fourier coefficients of the impulse train. The Fourier coefficients can be computed as
$
  c_k = & 1/T_s integral_(-T_s/2)^(T_s/2) p(t) e^(-i k omega_s t) dif t \
      = & 1/T_s integral_(-T_s/2)^(T_s/2) (sum_(n = -oo)^(oo) delta (t - n T_s)) e^(-i k omega_s t) dif t, \
$
since the integrand is non-zero only when $t = 0$ and $n = 0$, we have
$
  c_k = 1/T_s integral_(-T_s/2)^(T_s/2) delta (t) e^(-i k omega_s t) dif t = 1/T_s integral_(-T_s/2)^(T_s/2) delta (t) dif t = 1/T_s = f_s.
$
Hence, the impulse train can be expressed as
$
  p(t) = sum_(k = -oo)^(oo) f_s e^(i k omega_s t) = f_s sum_(k = -oo)^(oo) e^(i k omega_s t).
$
Now as we want to analyze the frequency content of the sampled signal, we can take the Fourier transform of the sampled signal $x_s(t) = x(t) p(t)$:
$
  cal(F){x_s(t)} = & cal(F){x(t) p(t)} \
     X_s (omega) = & integral_RR x(t) p(t) e^(-i omega t) dif t \
                 = & integral_RR x(t) (sum_(k = -oo)^(oo) f_s e^(i k omega_s t)) e^(-i omega t) dif t \
                 = & integral_RR (sum_(k = -oo)^(oo) f_s x(t) e^(-i (omega - k omega_s) t)) dif t \
                 = & f_s sum_(k = -oo)^(oo) integral_RR x(t) e^(-i (omega - k omega_s) t) dif t. \
$
Notice that the integral in the last line is just the Fourier transform of $x(t)$ evaluated at $omega - k omega_s$, so we can write
$
  X_s (omega) = f_s sum_(k = -oo)^(oo) X(omega - k omega_s). qed
$

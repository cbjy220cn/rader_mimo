function fenxi(sig,fs,T)


subplot(221);
plot(0:1/fs:T-1/fs,real(sig));
xlabel('时间(s)'); ylabel('实部幅度(v)');
title('FMCW 实部信号'); axis tight;grid on
subplot(222);
plot(0:1/fs:T-1/fs,imag(sig));
xlabel('时间(s)'); ylabel('虚部幅度(v)');
title('FMCW 虚部信号'); axis tight;grid on
subplot(223)
sp.p3(sig);
grid on;axis tight;
subplot(224);
spectrogram(sig,32,16,32,fs,'yaxis');
title('FMCW 信号时频分析');

end
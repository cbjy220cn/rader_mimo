function t = hadamard_decode(resp,H)
    [N1,N2,N3,N4] = size(resp);
    for i = 1:N1
        for j = 1:N2
           t(i,j,:) = squeeze(resp(i,j,:,:)) * H;
        end 
    end
end 
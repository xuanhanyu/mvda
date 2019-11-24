function Vs = sorted_eig(W, argmax)
[V, D] = eig(W);
if argmax
    [d, ind] = sort(abs(diag(D)),'descend');
else
    [d, ind] = sort(abs(diag(D)),'ascend');
end
Vs = V(:,ind);
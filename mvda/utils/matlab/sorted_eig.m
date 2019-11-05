function Vs = sorted_eig(W, argmax)
[V, D] = eig(W);
if argmax
    [d, ind] = sort(diag(D),'descend');
else
    [d, ind] = sort(diag(D),'ascend');
end
Vs = V(:,ind);
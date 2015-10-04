% Set K
clc
centroids=initial_centroids

K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

numn=size(X,1);

for i=1:5
    for j=1:K
        hh=j
        xxx=X(i,:)
        kkk=centroids(j,:)
        diff=X(i,:)-centroids(j,:)
        norm(j)=diff*diff'
        diff=[];
    end
    norm
    [M,L] = min(norm)
    idx(i)=L;
    
    norm=[];
end
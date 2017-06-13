for j=1:16
    subplot(16,1,j);
    plot(k,interictal_segment_1.data(j:j,1:1000))
    axis off
end
    
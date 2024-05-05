% Define the folder path
folderPath = 'Bearing1_1';

numFiles = 1400;
totalSamplesPerCycle = 2560;

total_acc_matrix = zeros(totalSamplesPerCycle, numFiles);


for i = 1:numFiles

    data = csvread(fullfile(folderPath, sprintf('acc_%05d.csv', i)));
    

    hor_acc = data(:, 5);
    vert_acc = data(:, 6);
    

    total_acc = sqrt(hor_acc.^2 + vert_acc.^2);
    

    % total_acc = (total_acc - mean(total_acc))/std(total_acc);
    
    total_acc_matrix(:, i) = total_acc;
end


%%

% total_acc_matrix = normalize(total_acc_matrix);
concatenated_signal = reshape(total_acc_matrix, [], 1);
concatenated_signal = normalize(concatenated_signal);

totalSamples = numFiles * totalSamplesPerCycle;
plot(1:totalSamples, concatenated_signal);
xlabel('Sample Index');
ylabel('Total Acceleration');
title('Concatenated Total Acceleration Signal');



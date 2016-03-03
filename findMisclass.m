function [misclass_probs] = findMisclass(classified_label,true_label,class1, class2)

%sat=+1
%virg=-1

%sat=+1
%versi=-1, etc

%virg=+1
%versi=-1, etc

misclass_1_counter=0;
class_1_val=1;
misclass_2_counter=0;
class_2_val=-1;

for k=1:length(true_label)
    classified=classified_label(k);
    true=true_label(k);
    
    if(true == class_1_val)
        if(sign(classified) ~= sign(true))
            misclass_1_counter = misclass_1_counter +1;
        end
    elseif(true == class_2_val)
        if(sign(classified) ~= sign(true))
            misclass_2_counter = misclass_2_counter+1;
        end
    end
end

total_misclass = misclass_1_counter + misclass_2_counter;
misclass_1_prob=misclass_1_counter / (length(classified_label) / 2);
misclass_2_prob=misclass_2_counter / (length(classified_label) / 2);
misclass_total_prob = total_misclass / length(classified_label);
misclass_probs=[misclass_1_prob misclass_2_prob misclass_total_prob];

% disp(sprintf('Trial Error Prob [%s %s total]: %.2f %.2f %.2f',class1, class2, misclass_1_prob,misclass_2_prob,misclass_total_prob));


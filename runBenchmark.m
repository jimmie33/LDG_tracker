
root_dir = '../tracker_benchmark_v1.0';


output_dir = './output/tre_meem';
use_color = true;
use_experts = true;

if ~exist(output_dir)
    mkdir(output_dir)
end
load('task.mat');
% h = waitbar(0,'run benchmark ope');
for i = 1:numel(task)
    input_dir = strrep(fullfile(task{i}.data_dir,'img'),'\','/');
    if ~exist(input_dir,'dir')
        warning([input_dir, ' does not exist'])
        continue
    end
    ext = task{i}.ext;
    results = {};
    name = [task{i}.data_name, '_MEEM.mat'];
    for j = 1:numel(task{i}.run)
        start_frame = task{i}.run{j}.startFrame;
        init_rect = task{i}.run{j}.initRect;
        anno_begin = task{i}.run{j}.annoBegin;
        results{j} = MEEMTrack(use_color, use_experts, input_dir,ext,init_rect,start_frame);
        results{j}.annoBegin = anno_begin;
    end    
    save(fullfile(output_dir,name),'results','-v7.3');
%     waitbar(i/numel(task))
end
% close(h)
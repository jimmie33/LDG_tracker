
root_dir = '../tracker_benchmark_v1.0';


%% for tre

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get tre running parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% D = dir(fullfile([root_dir,'\results\results_TRE_CVPR13'],'*_MS.mat'));
% file_list={D.name};
% 
% task=cell(numel(file_list),1);
% for i = 1:numel(file_list)
%     load(fullfile([root_dir,'\results\results_TRE_CVPR13'],file_list{i}));
%     data_name = file_list{i}(1:end-7);
%     task{i}.data_name = data_name;
%     task{i}.data_dir = [root_dir,'\data\',data_name];
%     task{i}.ext = 'jpg';
%     for j = 1:numel(results)
%         task{i}.run{j}.startFrame = results{j}.startFrame;
%         task{i}.run{j}.initRect = results{j}.res(1,:);
%         task{i}.run{j}.annoBegin = results{j}.annoBegin;
% %         task{i}.run{j}.anno = results{j}.anno;
%     end
% end
% 
% save('task.mat','task','-v7.3')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output_dir = './output/ope_meem';
if ~exist(output_dir)
    mkdir(output_dir)
end
load('task.mat');
% h = waitbar(0,'run benchmark ope');
for i = 17%1:numel(task)
    input_dir = strrep(fullfile(task{i}.data_dir,'img'),'\','/');
    if ~exist(input_dir,'dir')
        warning([input_dir, ' does not exist'])
        continue
    end
    ext = task{i}.ext;
    results = {};
    name = [task{i}.data_name, '_MEEM.mat'];
    for j = 1:1 %numel(task{i}.run)
        start_frame = task{i}.run{j}.startFrame;
        init_rect = task{i}.run{j}.initRect;
        anno_begin = task{i}.run{j}.annoBegin;
        results{j} = MEEMTrack(input_dir,ext,init_rect,start_frame);
        results{j}.annoBegin = anno_begin;
    end    
    save(fullfile(output_dir,name),'results','-v7.3');
%     waitbar(i/numel(task))
end
% close(h)
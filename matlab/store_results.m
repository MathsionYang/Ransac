folder = '../results/homography/';
folder_res = '../results/homography/';

% folder = '../results/line2d/';
% folder_res = '../results/line2d/';

% folder = '../results/kusvod2/';
% folder_res = '../results/kusvod2/';
% 
% folder = '../results/adelaidermf/';
% folder_res = '../results/adelaidermf/';

fileID = fopen([folder 'uniform_010_m.csv']);
t = textscan(fileID,'%s');
images = split (t{:}, ',');
images = images(:,1);

% read from 2 column
% uniform000 = csvread([folder 'uniform_000_m.csv'], 0, 1);
% uniform100 = csvread([folder 'uniform_100_m.csv'], 0, 1);
uniform010 = csvread([folder 'uniform_010_m.csv'], 0, 1);
uniform011 = csvread([folder 'uniform_011_m.csv'], 0, 1);
% uniform001 = csvread([folder 'uniform_001_m.csv'], 0, 1);
% uniform111 = csvread([folder 'uniform_111_m.csv'], 0, 1);

% prosac000 = csvread([folder 'prosac_000_m.csv'], 0, 1);
% prosac100 = csvread([folder 'prosac_100_m.csv'], 0, 1);
prosac010 = csvread([folder 'prosac_010_m.csv'], 0, 1);
prosac011 = csvread([folder 'prosac_011_m.csv'], 0, 1);
% prosac001 = csvread([folder 'prosac_001_m.csv'], 0, 1);
% prosac111 = csvread([folder 'prosac_111_m.csv'], 0, 1);

% napsac000 = csvread([folder 'napsac_000_m.csv'], 0, 1);
% napsac100 = csvread([folder 'napsac_100_m.csv'], 0, 1);
% napsac010 = csvread([folder 'napsac_010_m.csv'], 0, 1);
% napsac001 = csvread([folder 'napsac_001_m.csv'], 0, 1);
% napsac111 = csvread([folder 'napsac_111_m.csv'], 0, 1);


algorithms = {
            'uniform_gc', ...
            'uniform_gc_sprt', ...
            'prosac_gc', ...
            'prosac_gc_sprt'};
%             'prosac_sprt', ...
%             'prosac_lo_gc_sprt'};
%             'napsac', ...
%             'napsac_lo', ...
%             'napsac_gc', ...
%             'napsac_sprt', ...
%             'napsac_lo_gc_sprt'};

inl = 1;
it = 2;
lo_it = 3;
tm = 4;
er = 5;
fl = 6;
        
inliers = [uniform010(:,inl) uniform011(:,inl) ...
           prosac010(:,inl) prosac011(:,inl)];

iters =  [uniform010(:,it) uniform011(:,it) ...
          prosac010(:,it) prosac011(:,it)];
       
lo_iters =  [uniform010(:,lo_it) uniform010(:,lo_it) ...
             prosac011(:,lo_it) prosac011(:,lo_it)];

time = [uniform010(:,tm) uniform011(:,tm) ...
         prosac010(:,tm)  prosac011(:,tm)]; 

fails = [uniform010(:,fl) uniform011(:,fl) ...
          prosac010(:,fl)  prosac011(:,fl)];

fails2 = [uniform010(:,fl+1) uniform011(:,fl+1) ...
           prosac010(:,fl+1)  prosac011(:,fl+1)];

fails3 = [uniform010(:,fl+2) uniform011(:,fl+2) ...
           prosac010(:,fl+2)  prosac011(:,fl+2)];

errors = [uniform010(:,er) uniform011(:,er) ...
         prosac010(:,er) prosac011(:,er)];


num_images = numel (images);
num_algs = numel (algorithms);
alg = {'1' '2' '3' '4'}; %  '5' '6' '7' '8' '9' 'A'};

 
save_results (inliers, images, algorithms, alg, 'Average number of inliers', [folder_res 'inliers'])
save_results (time, images, algorithms, alg, 'Average time (mcs)', [folder_res 'time'])
save_results (iters, images, algorithms, alg, 'Average number of iterations', [folder_res 'iters'])
save_results (lo_iters, images, algorithms, alg, 'Average number of LO iterations', [folder_res 'lo_iters'])
save_results (fails, images, algorithms, alg, 'Number of fails < 10%', [folder_res 'fails1'])
save_results (fails2, images, algorithms, alg, 'Number of fails < 25%', [folder_res 'fails2'])
save_results (fails3, images, algorithms, alg, 'Number of fails < 50%', [folder_res 'fails3'])
save_results (errors, images, algorithms, alg, 'Average Error', [folder_res 'errors'])


% criteria = {'inliers', 'std inliers', 'iters', 'std iters', 'time', 'std time', 'fails'};
% for img = 1:numel (images)
%     for cr = 1:numel(criteria) 
%         subplot (numel(criteria), 1, cr);
%         data = [uniform000(img,cr) uniform100(img,cr) uniform010(img,cr) uniform001(img,cr) uniform111(img,cr)];
%         if (cr == numel(criteria))
%             h = heatmap(algorithms, criteria{cr}, data);
%         else
%             h = heatmap(alg, criteria{cr}, data);
%         end
%         if (cr == 1)
%             h.title(images{img});
%         end
%     end
%     print('-fillpage',[folder_res images{img}],'-dpdf', '-r300')
%     close
% end

function save_results (data, images, algorithms, alg, titl, save)
    figure('units','normalized','outerposition',[0 0 1 1])
    title ('asdad')

    subplot (numel (images)+3, 1, 1);
    h = heatmap(alg, 'title show', ones (size(data(1,:))));
    h.title(titl);

    for img = 1:numel (images)
        subplot (numel (images)+3, 1, img+1);
        h = heatmap(alg, images{img}, data(img,:));
    end
    
    subplot (numel (images)+3, 1, numel(images)+2)
    h = heatmap (alg, 'Average (all)', mean (data));
    
    subplot (numel (images)+3, 1, numel(images)+3)
    h = heatmap(algorithms, 'i', ones (size(data(1,:))));
    
    print (gcf, [save '.png'], '-dpng', '-r300');
%     print('-fillpage',save,'-dpdf', '-r300')
    close;
end

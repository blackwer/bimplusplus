function [] = make_movie(filename)
%MAKE_MOVIE Make a movie from bim_test data
load('alpha.mat', 'alpha')

window_x=1.2;
window_y=1.2;

fig = figure(); set(gca,'FontSize',18); set(gcf,'color','w');

%% -------- save plot to movie --------
VW = VideoWriter([filename '.avi']);
VW.FrameRate = 30;
open(VW);

for i = 1:99
    fname = 'test_' + string(i) + '.mat';
    load(fname, 'positions', 'thetas', 'U_n');
    t = 100*0.001*i;
    
    %% update theta and L
    normals = [-sin(thetas); cos(thetas)];
    area_n = 3.186673437718255;
    
    %% conserved quantities
    area_np2 = 0.5*trapzp(positions(1,:).^2+positions(2,:).^2, length(alpha)); % integral of r dr dtheta

    %% plot & save image
    clf;
    plot([positions(1,:) positions(1,1)],[positions(2,:) positions(2,1)],'bo-','LineWidth',2); grid on;
    axis([-window_x window_x -window_y window_y]); axis equal; hold on;
    plot(sqrt(area_n/pi)*[cos(alpha) cos(alpha(1))],sqrt(area_n/pi)*[sin(alpha) sin(alpha(1))],'r-');
    %plot(cm_np2(1),cm_np2(2),'r.','MarkerSize',10);
    title([' t = ' num2str(t) ' , ' 'A/A_0 = ' num2str(area_np2/area_n)]);
    q2 = quiver(positions(1,:),positions(2,:),U_n.*normals(1,:),U_n.*normals(2,:)); q2.AutoScale = 'off';
    frame = getframe(fig);
    writeVideo(VW,frame);
end
close(fig)

end


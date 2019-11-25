%% periodic trapezoid rule over full period w.r.t. alpha = s*2pi/L
function out = trapzp(vec, N)

out = 2*pi/N*sum(vec);

end
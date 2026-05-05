function [etaSol,F,Fp,Fpp,s] = solve_blasius_ode45_mod(etaMax)
%SOLVE_BLASIUS_ODE45_HUANG Solve F''' + F F'' = 0 by shooting.

    if nargin < 1
        etaMax = 10;
    end

    odeOpts = odeset('RelTol',1e-11,'AbsTol',1e-13);

    sVals = linspace(0.1,1.0,80);
    errVals = zeros(size(sVals));
    for i = 1:numel(sVals)
        errVals(i) = shoot_error(sVals(i),etaMax,odeOpts);
    end

    idx = find(errVals(1:end-1).*errVals(2:end) < 0,1);
    if isempty(idx)
        error('solve_blasius_ode45_huang:NoBracket', ...
              'Could not bracket F''''(0) on [0.1,1.0].');
    end

    s = fzero(@(q) shoot_error(q,etaMax,odeOpts), ...
              [sVals(idx),sVals(idx+1)]);

    [etaSol,Y] = ode45(@blasius_ode,[0 etaMax],[0;0;s],odeOpts);
    F = Y(:,1);
    Fp = Y(:,2);
    Fpp = Y(:,3);
end

function err = shoot_error(s,etaMax,odeOpts)
    [~,Y] = ode45(@blasius_ode,[0 etaMax],[0;0;s],odeOpts);
    err = Y(end,2) - 1;
end

function dY = blasius_ode(~,Y)
    dY = [Y(2); Y(3); -Y(1).*Y(3)];
end
function T = generate_blasius_streamfunction_data(nx,ny,Ue,nu,xRange,yRange,sampleMode)
%GENERATE_BLASIUS_HUANG_DATA Generate one Blasius streamfunction dataset.
%
% sampleMode is 'uniform' or 'random'. The table contains only measured
% coordinates and fields; exact jets are added separately for diagnostics.

    if nargin < 1, nx = 36; end
    if nargin < 2, ny = 36; end
    if nargin < 3, Ue = 1.0; end
    if nargin < 4, nu = 1e-2; end
    if nargin < 5, xRange = [0.5 2.0]; end
    if nargin < 6, yRange = [0.0 0.8]; end
    if nargin < 7, sampleMode = 'uniform'; end

    switch lower(sampleMode)
        case 'uniform'
            xVec = linspace(xRange(1),xRange(2),nx);
            yVec = linspace(yRange(1),yRange(2),ny);
            [X,Y] = meshgrid(xVec,yVec);
            x = X(:);
            y = Y(:);
        case 'random'
            n = nx*ny;
            x = xRange(1) + diff(xRange).*rand(n,1);
            y = yRange(1) + diff(yRange).*rand(n,1);
        otherwise
            error('generate_blasius_huang_data:BadSampleMode', ...
                  'sampleMode must be ''uniform'' or ''random''.');
    end

    eta = y.*sqrt(Ue./(2*nu*x));
    etaMax = max(10,1.05*max(eta));
    [etaSol,F,Fp,~,~] = solve_blasius_ode45_mod(etaMax);

    Ffun = @(q) interp1(etaSol,F,q,'pchip','extrap');
    Fpfun = @(q) interp1(etaSol,Fp,q,'pchip','extrap');

    Fvals = Ffun(eta);
    Fpvals = Fpfun(eta);

    psi = sqrt(2*nu*Ue*x).*Fvals;
    u = Ue.*Fpvals;
    v = sqrt(nu*Ue./(2*x)).*(eta.*Fpvals - Fvals);

    T = table(x,y,eta,psi,u,v,Ue*ones(size(x)),nu*ones(size(x)), ...
              'VariableNames',{'x','y','eta','psi','u','v','Ue','nu'});
end
function [Paug,meta] = build_Ptilde_blasius_boundary_control(Tjet,nu,opts)
%BUILD_PTILDE_AUGMENTED_BLASIUS_SAFE Optional known-PDE plus wall rows.
%
% This is a boundary-value diagnostic, not the paper's unknown-PDE normal
% construction. Coefficients returned by the SVD live in normalized
% coordinates when ColumnNormalize is true; compare c_true after multiplying
% by meta.colNorms.

    if nargin < 2 || isempty(nu)
        nu = Tjet.nu(1);
    end
    if nargin < 3, opts = struct(); end

    rawOpts = opts;
    rawOpts.RowNormalize = false;
    rawOpts.ColumnNormalize = false;
    [~,pdeMeta] = build_Ptilde_from_known_blasius_pde(Tjet,nu,rawOpts);
    Ppde = pdeMeta.Praw;

    x = Tjet.x(:);
    y = Tjet.y(:);
    psi = Tjet.psi(:);
    wall = abs(y) <= wall_tolerance(y);

    xw = x(wall);
    yw = y(wall);
    psw = psi(wall);
    nw = numel(xw);

    PwallZeta = zeros(nw,12);
    PwallZeta(:,5) = 1;
    PwallZeta(:,6) = xw;
    PwallZeta(:,7) = yw;
    PwallZeta(:,8) = psw;

    PwallPhi = zeros(nw,12);
    PwallPhi(:,9) = 1;
    PwallPhi(:,10) = xw;
    PwallPhi(:,11) = yw;
    PwallPhi(:,12) = psw;

    Praw = [Ppde; PwallZeta; PwallPhi];
    [Paug,rowNorms,colNorms] = normalize_matrix(Praw,opts);

    meta = struct();
    meta.Praw = Praw;
    meta.rowNorms = rowNorms;
    meta.colNorms = colNorms;
    meta.nPdeRows = size(Ppde,1);
    meta.nWallRows = 2*nw;
end

function tol = wall_tolerance(y)
    yr = max(y) - min(y);
    tol = max(1e-12,1e-10*max(1,yr));
end

function [P,rowNorms,colNorms] = normalize_matrix(Praw,opts)
    rowNormalize = true;
    colNormalize = true;
    if isfield(opts,'RowNormalize'), rowNormalize = opts.RowNormalize; end
    if isfield(opts,'ColumnNormalize'), colNormalize = opts.ColumnNormalize; end

    P = Praw;
    rowNorms = ones(size(P,1),1);
    if rowNormalize
        rowNorms = vecnorm(P,2,2);
        rowNorms(rowNorms == 0) = 1;
        P = bsxfun(@rdivide,P,rowNorms);
    end

    colNorms = ones(1,size(P,2));
    if colNormalize
        colNorms = vecnorm(P,2,1);
        colNorms(colNorms == 0) = 1;
        P = bsxfun(@rdivide,P,colNorms);
    end
end
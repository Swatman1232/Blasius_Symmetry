function [Ppde,meta] = build_Ptilde_from_known_blasius_pde(Tjet,nu,opts)
%BUILD_PTILDE_BLASIUS_PDE_SAFE Known-PDE invariance matrix for diagnostics.
%
% This is not the Huang unknown-dynamics normal construction. It is a
% control check using grad(F), where
% F = psi_y*psi_xy - psi_x*psi_yy - nu*psi_yyy.

    if nargin < 2 || isempty(nu)
        nu = Tjet.nu(1);
    end
    if nargin < 3, opts = struct(); end

    Z = [Tjet.x(:),Tjet.y(:),Tjet.psi(:), ...
         Tjet.psi_x(:),Tjet.psi_y(:), ...
         Tjet.psi_xx(:),Tjet.psi_xy(:),Tjet.psi_yy(:), ...
         Tjet.psi_xxx(:),Tjet.psi_xxy(:),Tjet.psi_xyy(:),Tjet.psi_yyy(:)];

    nPts = size(Z,1);
    pointMask = point_mask(opts,nPts);
    keepIdx = find(pointMask);
    Praw = zeros(numel(keepIdx),12);

    for row = 1:numel(keepIdx)
        i = keepIdx(row);
        ux = Z(i,4);
        uy = Z(i,5);
        uxy = Z(i,7);
        uyy = Z(i,8);

        gradF = zeros(1,12);
        gradF(4) = -uyy;
        gradF(5) = uxy;
        gradF(7) = uy;
        gradF(8) = -ux;
        gradF(12) = -nu;

        L = blasius_prolongation_matrix_order3(Z(i,:));
        Praw(row,:) = gradF * L;
    end

    [Ppde,rowNorms,colNorms] = normalize_matrix(Praw,opts);

    meta = struct();
    meta.Praw = Praw;
    meta.rowNorms = rowNorms;
    meta.colNorms = colNorms;
    meta.pointMask = pointMask;
    meta.nPointsUsed = numel(keepIdx);
end

function mask = point_mask(opts,nPts)
    mask = true(nPts,1);
    if isfield(opts,'PointMask')
        mask = logical(opts.PointMask(:));
        if numel(mask) ~= nPts
            error('build_Ptilde_blasius_pde_safe:BadPointMask', ...
                  'PointMask must have one entry per point.');
        end
    end
end

function [P,rowNorms,colNorms] = normalize_matrix(Praw,opts)
    rowNormalize = false;
    colNormalize = false;
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

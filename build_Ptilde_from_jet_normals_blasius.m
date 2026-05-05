function [Ptilde,meta] = build_Ptilde_from_jet_normals_blasius(Tjet,normals,opts)
%BUILD_PTILDE_BLASIUS_NORMALS_HUANG Assemble Huang-style normal matrix.
%
% normals must be Dp x (Dp-d) x N, usually info.normals3 from
% gmls_prolongate_psi_order3_huang.

    if nargin < 3, opts = struct(); end

    Z = jet_matrix(Tjet);
    nPts = size(Z,1);
    codim = size(normals,2);
    pointMask = point_mask(opts,nPts);
    keepIdx = find(pointMask);

    Praw = zeros(numel(keepIdx)*codim,12);
    row = 0;

    for ii = 1:numel(keepIdx)
        i = keepIdx(ii);
        L = blasius_prolongation_matrix_order3(Z(i,:));
        for q = 1:codim
            row = row + 1;
            Praw(row,:) = normals(:,q,i)' * L;
        end
    end

    [Ptilde,rowNorms,colNorms] = normalize_matrix(Praw,opts);

    meta = struct();
    meta.Praw = Praw;
    meta.rowNorms = rowNorms;
    meta.colNorms = colNorms;
    meta.codim = codim;
    meta.pointMask = pointMask;
    meta.nPointsUsed = numel(keepIdx);
end

function Z = jet_matrix(Tjet)
    Z = [Tjet.x(:),Tjet.y(:),Tjet.psi(:), ...
         Tjet.psi_x(:),Tjet.psi_y(:), ...
         Tjet.psi_xx(:),Tjet.psi_xy(:),Tjet.psi_yy(:), ...
         Tjet.psi_xxx(:),Tjet.psi_xxy(:),Tjet.psi_xyy(:),Tjet.psi_yyy(:)];
end

function mask = point_mask(opts,nPts)
    mask = true(nPts,1);
    if isfield(opts,'PointMask')
        mask = logical(opts.PointMask(:));
        if numel(mask) ~= nPts
            error('build_Ptilde_blasius_normals_huang:BadPointMask', ...
                  'PointMask must have one entry per point.');
        end
    end
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

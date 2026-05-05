function [Pcombo,meta] = build_Ptilde_blasius_freestream_control(Tjet,nu,opts)
% PDE rows plus fixed-free-stream rows phi_y = 0 near top boundary.

    if nargin < 2 || isempty(nu)
        nu = Tjet.nu(1);
    end
    if nargin < 3
        opts = struct();
    end

    rawOpts = opts;
    rawOpts.RowNormalize = false;
    rawOpts.ColumnNormalize = false;

    [~,pdeMeta] = build_Ptilde_from_known_blasius_pde(Tjet,nu,rawOpts);
    PpdeRaw = pdeMeta.Praw;

    PffRaw = freestream_rows(Tjet,opts);

    Praw = [PpdeRaw; PffRaw];

    [Pcombo,rowNorms,colNorms] = normalize_matrix(Praw,opts);

    meta = struct();
    meta.Praw = Praw;
    meta.PpdeRaw = PpdeRaw;
    meta.PffRaw = PffRaw;
    meta.rowNorms = rowNorms;
    meta.colNorms = colNorms;
end

function Pff = freestream_rows(Tjet,opts)

    frac = 0.05;
    if isfield(opts,'FarfieldFraction')
        frac = opts.FarfieldFraction;
    end

    Z = [Tjet.x(:),Tjet.y(:),Tjet.psi(:), ...
         Tjet.psi_x(:),Tjet.psi_y(:), ...
         Tjet.psi_xx(:),Tjet.psi_xy(:),Tjet.psi_yy(:), ...
         Tjet.psi_xxx(:),Tjet.psi_xxy(:),Tjet.psi_xyy(:),Tjet.psi_yyy(:)];

    y = Tjet.y(:);
    yMax = max(y);
    yRange = max(y) - min(y);

    far = y >= yMax - frac*yRange;
    keepIdx = find(far);

    Pff = zeros(numel(keepIdx),12);

    for r = 1:numel(keepIdx)
        i = keepIdx(r);
        L = blasius_prolongation_matrix_order3(Z(i,:));

        % Row 5 is the prolongation coefficient for psi_y.
        % Fixed Ue means psi_y = Ue should be preserved, so phi_y = 0.
        Pff(r,:) = L(5,:);
    end
end

function [P,rowNorms,colNorms] = normalize_matrix(Praw,opts)

    rowNormalize = true;
    colNormalize = true;

    if isfield(opts,'RowNormalize')
        rowNormalize = opts.RowNormalize;
    end
    if isfield(opts,'ColumnNormalize')
        colNormalize = opts.ColumnNormalize;
    end

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
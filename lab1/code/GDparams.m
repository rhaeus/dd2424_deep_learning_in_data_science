classdef GDparams
    
    properties
        n_batch
        eta
        n_epochs
    end
    
    methods
        function obj = GDparams(n_batch,eta,n_epochs)
            obj.n_batch = n_batch;
            obj.eta = eta;
            obj.n_epochs = n_epochs;
        end
        
    end
end


function tracker=particleTracker()
tracker.state=zeros(1000,4);
tracker.weight=ones(1000,1);
tracker.radius=0.5;
tracker.output=zeros(1,4);
tracker.temp=zeros(1);
tracker.topK=zeros(10,4);

tracker.ln_rate = 0.05;
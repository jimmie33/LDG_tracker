function tracker=particleTracker()
tracker.state=zeros(500,4);
tracker.weight=ones(500,1);
tracker.radius=0.25;
tracker.output=zeros(1,4);
tracker.temp=zeros(1);
tracker.topK=zeros(10,4);

tracker.ln_rate = 0.05;